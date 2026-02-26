from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sqlite3
import hashlib
import json
import sys
import queue
import threading
import asyncio
import time
from datetime import datetime
import os

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.append(r'C:\Users\LENOVO\OneDrive')

# ── Import Original AI Planner ────────────────────────────────────────────────
try:
    from ai_planner import process_plan_for_api, EnhancedMultiAIPlanner as _OldPlanner
    AI_PLANNER_AVAILABLE = True
    print("✅ AI Planner (original) imported successfully!")
except Exception as e:
    AI_PLANNER_AVAILABLE = False
    _OldPlanner = None
    print(f"⚠️  AI Planner import failed: {e}")

# ── Import V3 Planner ─────────────────────────────────────────────────────────
try:
    from planner_v3 import EnhancedMultiAIPlannerV3
    V3_AVAILABLE = True
    print("✅ Planner V3 imported successfully!")
except Exception as e:
    V3_AVAILABLE = False
    print(f"⚠️  Planner V3 import failed: {e}")

# ── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Active SSE connections tracker
active_streams = {}

# All 12 agents list
ALL_AGENTS = [
    "gemini", "openai", "together", "groq", "replicate",
    "fireworks", "cohere", "deepseek", "openrouter",
    "writer", "huggingface", "pawn"
]


# ── Helper: Build V3 Planner Instance ────────────────────────────────────────
def build_v3_planner(sse_emit_fn=None):
    """
    Creates a fresh EnhancedMultiAIPlannerV3 instance by wrapping the
    original planner's methods. Returns None if unavailable.
    """
    if not AI_PLANNER_AVAILABLE or not V3_AVAILABLE:
        return None
    try:
        _old = _OldPlanner()
        return EnhancedMultiAIPlannerV3(
            call_ai_api_fn           = _old.call_ai_api,
            create_scoring_prompt_fn = _old.create_scoring_prompt,
            fraud_check_fn           = _old.check_fraud_and_misinformation,
            feature_detect_fn        = _old.detect_features_from_mini_plan,
            sse_emit_fn              = sse_emit_fn,
        )
    except Exception as e:
        print(f"⚠️  Could not build V3 planner: {e}")
        return None


# ── Helper: Run Async in New Event Loop ──────────────────────────────────────
def run_async(coro):
    """Safely run an async coroutine from a sync/thread context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Database Setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('planner.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            aim TEXT NOT NULL,
            steps TEXT NOT NULL,
            result TEXT,
            features TEXT,
            competition_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_db():
    conn = sqlite3.connect('planner.db')
    conn.row_factory = sqlite3.Row
    return conn


# ═════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data     = request.json
        username = data.get('username')
        email    = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400

        hashed = hash_password(password)
        conn   = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed)
            )
            conn.commit()
            conn.close()
            print(f"✅ New user registered: {username}")
            return jsonify({'success': True, 'message': 'User created successfully'}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 409

    except Exception as e:
        print(f"❌ Signup error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data     = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400

        hashed = hash_password(password)
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT username, email FROM users WHERE username = ? AND password = ?',
            (username, hashed)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            print(f"✅ User logged in: {username}")
            return jsonify({'success': True, 'username': user['username'], 'email': user['email']}), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ═════════════════════════════════════════════════════════════════════════════
# STREAMING PLAN ROUTE  (V3 planner with SSE)
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/plan/stream', methods=['POST'])
def plan_stream():
    """Real-time plan generation via Server-Sent Events — powered by V3 planner."""
    try:
        data     = request.json
        username = data.get('username')
        aim      = data.get('aim')
        steps    = data.get('steps')

        if not username or not aim or not steps:
            return jsonify({'success': False, 'message': 'Username, aim, and steps are required'}), 400

        print(f"\n{'='*60}")
        print(f"🔴 STREAMING (V3): New request from: {username}")
        print(f"🎯 Aim: {aim}")
        print(f"📋 Steps count: {len(steps)}")
        print(f"{'='*60}\n")

        event_queue = queue.Queue()
        session_id  = f"{username}_{int(time.time())}"
        active_streams[session_id] = event_queue

        # Save plan record to DB
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO plans (username, aim, steps) VALUES (?, ?, ?)',
            (username, aim, json.dumps(steps))
        )
        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()

        def generate():
            """SSE generator — streams events from event_queue to client."""
            try:
                yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id, 'planner': 'v3'})}\n\n"

                def run_planning():
                    try:
                        print(f"🤖 V3 planner starting for session: {session_id}")

                        # SSE emit function — pushes events into the queue
                        def sse_emit(msg, t="info"):
                            event_queue.put({
                                'type': t,
                                'message': msg,
                                'timestamp': datetime.now().isoformat()
                            })

                        # ── Try V3 first ──────────────────────────────────
                        v3 = build_v3_planner(sse_emit_fn=sse_emit)

                        if v3:
                            result = run_async(
                                v3.run_full_planning_session(
                                    aim           = aim,
                                    initial_steps = steps,
                                    parameters    = [],
                                    agents        = ALL_AGENTS,
                                )
                            )
                        else:
                            # ── Fallback to original planner ─────────────
                            print("⚠️  V3 unavailable — falling back to original planner")
                            result = process_plan_for_api(username, aim, steps, event_queue)

                        # Save result to DB
                        competition_data = result.get('competition_analysis', {})
                        conn   = get_db()
                        cursor = conn.cursor()
                        cursor.execute(
                            'UPDATE plans SET result = ?, competition_data = ? WHERE id = ?',
                            (json.dumps(result), json.dumps(competition_data), plan_id)
                        )
                        conn.commit()
                        conn.close()

                        # Push final result event
                        event_queue.put({
                            'type':      'final_result',
                            'data':      result,
                            'timestamp': datetime.now().isoformat()
                        })
                        event_queue.put(None)  # completion signal
                        print(f"✅ Planning complete for session: {session_id}")

                    except Exception as e:
                        print(f"❌ Planning error [{session_id}]: {e}")
                        import traceback; traceback.print_exc()
                        event_queue.put({
                            'type':      'error',
                            'data':      {'message': str(e)},
                            'timestamp': datetime.now().isoformat()
                        })
                        event_queue.put(None)

                # Run planning in background thread (keeps Flask SSE alive)
                t = threading.Thread(target=run_planning, daemon=True)
                t.start()

                # Stream events to client
                heartbeat = 0
                while True:
                    try:
                        event = event_queue.get(timeout=30)
                        if event is None:
                            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                            break
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        heartbeat += 1
                        yield f"data: {json.dumps({'type': 'heartbeat', 'count': heartbeat})}\n\n"

            except GeneratorExit:
                print(f"⚠️  Client disconnected: {session_id}")
            except Exception as e:
                print(f"❌ Stream error [{session_id}]: {e}")
                import traceback; traceback.print_exc()
            finally:
                active_streams.pop(session_id, None)
                print(f"🧹 Session cleaned up: {session_id}")

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control':           'no-cache',
                'X-Accel-Buffering':       'no',
                'Connection':              'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        print(f"❌ Stream endpoint error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# STANDARD (NON-STREAMING) PLAN ROUTE  (V3 planner)
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/plan', methods=['POST'])
def plan():
    """Standard JSON planning endpoint — powered by V3 planner."""
    try:
        data     = request.json
        username = data.get('username')
        aim      = data.get('aim')
        steps    = data.get('steps')

        if not username or not aim or not steps:
            return jsonify({'success': False, 'message': 'Username, aim, and steps are required'}), 400

        print(f"\n{'='*60}")
        print(f"🔥 PLAN (V3): Request from: {username}")
        print(f"🎯 Aim: {aim}")
        print(f"{'='*60}\n")

        # Save to DB
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO plans (username, aim, steps) VALUES (?, ?, ?)',
            (username, aim, json.dumps(steps))
        )
        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()

        if not AI_PLANNER_AVAILABLE:
            return jsonify({'success': False, 'message': 'AI planner not available'}), 503

        try:
            # ── Try V3 first ──────────────────────────────────────────────
            v3 = build_v3_planner()
            if v3:
                result = run_async(
                    v3.run_full_planning_session(
                        aim           = aim,
                        initial_steps = steps,
                        parameters    = [],
                        agents        = ALL_AGENTS,
                    )
                )
            else:
                # ── Fallback to original ──────────────────────────────────
                print("⚠️  V3 unavailable — using original planner")
                result = process_plan_for_api(username, aim, steps)

            if "error" in result:
                return jsonify({'success': False, 'message': f"AI error: {result['error']}"}), 500

            # Save result
            competition_data = result.get('competition_analysis', {})
            conn   = get_db()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE plans SET result = ?, competition_data = ? WHERE id = ?',
                (json.dumps(result), json.dumps(competition_data), plan_id)
            )
            conn.commit()
            conn.close()

            print(f"✅ Planning complete for {username}")
            return jsonify(result), 200

        except Exception as e:
            print(f"❌ Planner error: {e}")
            import traceback; traceback.print_exc()
            return jsonify({'success': False, 'message': f'AI processing error: {str(e)}'}), 500

    except Exception as e:
        print(f"❌ /api/plan error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ═════════════════════════════════════════════════════════════════════════════
# USER PLANS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/plans/<username>', methods=['GET'])
def get_plans(username):
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM plans WHERE username = ? ORDER BY created_at DESC',
            (username,)
        )
        plans = cursor.fetchall()
        conn.close()

        plans_list = []
        for p in plans:
            plan_data = {
                'id':         p['id'],
                'aim':        p['aim'],
                'steps':      json.loads(p['steps']),
                'created_at': p['created_at']
            }
            if p['result']:
                plan_data['result']          = json.loads(p['result'])
            if p['features']:
                plan_data['features']        = json.loads(p['features'])
            if p['competition_data']:
                plan_data['competition_data']= json.loads(p['competition_data'])
            plans_list.append(plan_data)

        return jsonify({'success': True, 'plans': plans_list}), 200

    except Exception as e:
        print(f"❌ Get plans error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ═════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE PARSING
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/parse-natural-language', methods=['POST'])
def parse_natural_language():
    try:
        data       = request.json
        user_input = data.get('input', '').strip()

        if not user_input:
            return jsonify({'success': False, 'message': 'Input text is required'}), 400
        if len(user_input) < 10:
            return jsonify({'success': False, 'message': 'Input too short'}), 400

        if not AI_PLANNER_AVAILABLE:
            return jsonify({'success': False, 'message': 'AI planner not available'}), 503

        try:
            from ai_planner import EnhancedMultiAIPlanner
            planner = EnhancedMultiAIPlanner()
            result  = planner.parse_natural_language_to_structure(user_input)

            if "error" in result:
                return jsonify({'success': False, 'message': result['error']}), 500

            print(f"✅ NLP parsed: AIM={result['aim'][:60]}, steps={len(result['steps'])}")
            return jsonify({
                'success':      True,
                'aim':          result['aim'],
                'steps':        result['steps'],
                'raw_response': result.get('raw_response', '')
            }), 200

        except Exception as e:
            print(f"❌ NLP error: {e}")
            import traceback; traceback.print_exc()
            return jsonify({'success': False, 'message': f'Parsing error: {str(e)}'}), 500

    except Exception as e:
        print(f"❌ /api/parse-natural-language error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ═════════════════════════════════════════════════════════════════════════════
# TOOL PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/process-tool', methods=['POST'])
def process_tool():
    try:
        data      = request.json
        tool_name = data.get('tool_name', '').strip()

        if not tool_name:
            return jsonify({'success': False, 'message': 'Tool name is required'}), 400
        if len(tool_name) < 3:
            return jsonify({'success': False, 'message': 'Tool name must be at least 3 characters'}), 400

        if not AI_PLANNER_AVAILABLE:
            return jsonify({'success': False, 'message': 'AI planner not available'}), 503

        print(f"\n🔧 Processing tool: {tool_name}")

        try:
            from ai_planner import EnhancedMultiAIPlanner
            planner = EnhancedMultiAIPlanner()
            result  = planner.process_software_tool_complete(tool_name)

            if result.get('success'):
                print(f"✅ Tool processed: {result['tools_found']} tools, {result['parameters_found']} params")
                return jsonify({
                    'success':          True,
                    'tools_found':      result['tools_found'],
                    'parameters_found': result['parameters_found'],
                    'tool_path':        result['tool_path'],
                    'param_path':       result['param_path']
                }), 200
            else:
                return jsonify({'success': False, 'message': result.get('error', 'Processing failed')}), 500

        except Exception as e:
            print(f"❌ Tool processing error: {e}")
            import traceback; traceback.print_exc()
            return jsonify({'success': False, 'message': str(e)}), 500

    except Exception as e:
        print(f"❌ /api/process-tool error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


@app.route('/api/remove-tool', methods=['POST'])
def remove_tool():
    try:
        data       = request.json
        tool_path  = data.get('tool_path',  r"C:\Users\LENOVO\Desktop\tool_name.xlsx")
        param_path = data.get('param_path', r"C:\Users\LENOVO\Desktop\parameter.xlsx")

        from openpyxl import Workbook

        for path, sheet_name, headers in [
            (tool_path,  "Tools",      ['Tool Name', 'Category', 'Function', 'Sub-options', 'Example']),
            (param_path, "Parameters", ['Tool Name', 'Parameter', 'Description']),
        ]:
            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name
            ws.append(headers)
            wb.save(path)
            print(f"✅ Cleared/created: {path}")

        return jsonify({'success': True, 'message': 'Tool removed and data cleared'}), 200

    except Exception as e:
        print(f"❌ Remove tool error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# COMPETITION ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/competition/<int:plan_id>', methods=['GET'])
def get_competition_data(plan_id):
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT competition_data, aim, username FROM plans WHERE id = ?',
            (plan_id,)
        )
        plan = cursor.fetchone()
        conn.close()

        if not plan:
            return jsonify({'success': False, 'message': 'Plan not found'}), 404
        if not plan['competition_data']:
            return jsonify({'success': False, 'message': 'No competition data available'}), 404

        return jsonify({
            'success':    True,
            'plan_id':    plan_id,
            'aim':        plan['aim'],
            'username':   plan['username'],
            'competition': json.loads(plan['competition_data'])
        }), 200

    except Exception as e:
        print(f"❌ Competition data error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ═════════════════════════════════════════════════════════════════════════════

def _admin_check():
    """Returns True if request carries valid admin key."""
    return request.headers.get('X-Admin-Key') == 'admin123'


@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    if not _admin_check():
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, username, email, created_at FROM users ORDER BY created_at DESC'
        )
        users = [dict(u) for u in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'users': users}), 200
    except Exception as e:
        print(f"❌ Admin users error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


@app.route('/api/admin/plans', methods=['GET'])
def get_all_plans():
    if not _admin_check():
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM plans ORDER BY created_at DESC LIMIT 50')
        plans_list = []
        for p in cursor.fetchall():
            plans_list.append({
                'id':              p['id'],
                'username':        p['username'],
                'aim':             p['aim'],
                'steps':           json.loads(p['steps']),
                'created_at':      p['created_at'],
                'has_result':      p['result'] is not None,
                'has_features':    p['features'] is not None,
                'has_competition': p['competition_data'] is not None
            })
        conn.close()
        return jsonify({'success': True, 'plans': plans_list}), 200
    except Exception as e:
        print(f"❌ Admin plans error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


@app.route('/api/admin/stats', methods=['GET'])
def get_stats():
    if not _admin_check():
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        conn   = get_db()
        cursor = conn.cursor()

        total_users     = cursor.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        total_plans     = cursor.execute('SELECT COUNT(*) FROM plans').fetchone()[0]
        completed_plans = cursor.execute('SELECT COUNT(*) FROM plans WHERE result IS NOT NULL').fetchone()[0]
        conn.close()

        return jsonify({
            'success': True,
            'stats': {
                'total_users':     total_users,
                'total_plans':     total_plans,
                'completed_plans': completed_plans,
                'pending_plans':   total_plans - completed_plans
            }
        }), 200
    except Exception as e:
        print(f"❌ Admin stats error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


@app.route('/api/admin/competition-stats', methods=['GET'])
def get_competition_stats():
    if not _admin_check():
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        conn   = get_db()
        cursor = conn.cursor()

        plans_with_comp = cursor.execute(
            'SELECT COUNT(*) FROM plans WHERE competition_data IS NOT NULL'
        ).fetchone()[0]

        rows = cursor.execute(
            'SELECT competition_data FROM plans WHERE competition_data IS NOT NULL'
        ).fetchall()
        conn.close()

        total_aim, total_step, total_clusters = 0, 0, 0
        for row in rows:
            try:
                c = json.loads(row['competition_data'])
                if c.get('aim_level'):
                    total_aim   += len(c['aim_level'].get('aim_competitors', []))
                if c.get('step_level'):
                    for s in c['step_level']:
                        total_step += s.get('total_companies', 0)
                if c.get('summary_level'):
                    total_clusters += len(c['summary_level'].get('cluster_analysis', []))
            except Exception:
                continue

        denom = max(plans_with_comp, 1)
        return jsonify({
            'success': True,
            'stats': {
                'plans_with_competition':       plans_with_comp,
                'total_aim_competitors':        total_aim,
                'total_step_competitors':       total_step,
                'total_clusters_analyzed':      total_clusters,
                'avg_aim_competitors_per_plan': round(total_aim  / denom, 2),
                'avg_step_competitors_per_plan':round(total_step / denom, 2)
            }
        }), 200
    except Exception as e:
        print(f"❌ Competition stats error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ── V3-specific admin route: memory bank stats ───────────────────────────────
@app.route('/api/admin/memory-stats', methods=['GET'])
def get_memory_stats():
    if not _admin_check():
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        v3 = build_v3_planner()
        if not v3:
            return jsonify({'success': False, 'message': 'V3 planner not available'}), 503
        return jsonify({'success': True, 'memory': v3.get_memory_stats()}), 200
    except Exception as e:
        print(f"❌ Memory stats error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':               'OK',
        'ai_planner_available': AI_PLANNER_AVAILABLE,
        'v3_available':         V3_AVAILABLE,
        'active_streams':       len(active_streams),
        'timestamp':            datetime.now().isoformat()
    }), 200


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    print("\n" + "="*65)
    print("🚀  AI PLANNER BACKEND — V3 EDITION")
    print("="*65)
    print(f"📊  Database      : planner.db")
    print(f"🤖  Original AI   : {'✅ Available' if AI_PLANNER_AVAILABLE else '⚠️  Not available'}")
    print(f"⚡  V3 Planner    : {'✅ Available' if V3_AVAILABLE else '⚠️  Not available'}")
    print(f"🌐  Server        : http://localhost:5000")
    print("="*65)
    print("📍  ENDPOINTS:")
    print("   POST  /api/signup                    → Register")
    print("   POST  /api/login                     → Login")
    print("   POST  /api/plan/stream               → SSE real-time planning (V3)")
    print("   POST  /api/plan                      → JSON planning (V3)")
    print("   POST  /api/parse-natural-language    → NLP → AIM + Steps")
    print("   POST  /api/process-tool              → Software tool extractor")
    print("   POST  /api/remove-tool               → Clear tool Excel files")
    print("   GET   /api/plans/<username>          → User plan history")
    print("   GET   /api/competition/<plan_id>     → Competition data")
    print("   GET   /api/admin/stats               → Admin stats")
    print("   GET   /api/admin/users               → Admin user list")
    print("   GET   /api/admin/plans               → Admin plan list")
    print("   GET   /api/admin/competition-stats   → Aggregate competition")
    print("   GET   /api/admin/memory-stats        → V3 memory bank stats")
    print("   GET   /health                        → Health check")
    print("="*65)
    if V3_AVAILABLE:
        print("⚡  V3 FEATURES ACTIVE:")
        print("   ✅ Adaptive Execution Graph (parallel agents)")
        print("   ✅ Cognitive Memory Bank (cross-session learning)")
        print("   ✅ Token Budget Controller (cost governance)")
        print("   ✅ BFT Consensus Engine (peer-review winner)")
        print("   ✅ Genetic Plan Mutator (evolutionary improvement)")
    print("="*65 + "\n")

    app.run(
        host        = "0.0.0.0",
        port        = 5000,
        debug       = False,
        use_reloader= False,
        threaded    = True
    )
