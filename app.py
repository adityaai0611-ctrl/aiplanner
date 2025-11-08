from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sqlite3
import hashlib
import json
import sys
import queue
import threading
import time
from datetime import datetime
import os

# Add the path to your ai_planner.py
sys.path.append(r'C:\Users\LENOVO\OneDrive')

# Import your AI planner
try:
    from ai_planner import process_plan_for_api
    AI_PLANNER_AVAILABLE = True
    print("✅ AI Planner imported successfully!")
except Exception as e:
    AI_PLANNER_AVAILABLE = False
    print(f"⚠️ AI Planner import failed: {e}")

app = Flask(__name__)
CORS(app)

# Store active SSE connections
active_streams = {}

# Database initialization
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

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_db():
    conn = sqlite3.connect('planner.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        
        hashed_password = hash_password(password)
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, hashed_password))
            conn.commit()
            conn.close()
            print(f"✅ New user registered: {username}")
            return jsonify({'success': True, 'message': 'User created successfully'}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 409
            
    except Exception as e:
        print(f"❌ Signup error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400
        
        hashed_password = hash_password(password)
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('SELECT username, email FROM users WHERE username = ? AND password = ?',
                      (username, hashed_password))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            print(f"✅ User logged in: {username}")
            return jsonify({'success': True, 'username': user['username'], 'email': user['email']}), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/plan/stream', methods=['POST'])
def plan_stream():
    """Stream real-time plan generation updates via Server-Sent Events"""
    try:
        data = request.json
        username = data.get('username')
        aim = data.get('aim')
        steps = data.get('steps')
        
        if not username or not aim or not steps:
            return jsonify({'success': False, 'message': 'Username, aim, and steps are required'}), 400
        
        print(f"\n{'='*60}")
        print(f"🔴 STREAMING: New planning request from: {username}")
        print(f"🎯 Aim: {aim}")
        print(f"📋 Steps: {steps}")
        print(f"{'='*60}\n")
        
        # Create event queue for this request
        event_queue = queue.Queue()
        session_id = f"{username}_{int(time.time())}"
        active_streams[session_id] = event_queue
        
        # Save to database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO plans (username, aim, steps) VALUES (?, ?, ?)',
                      (username, aim, json.dumps(steps)))
        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        def generate():
            """Generator function for SSE streaming"""
            try:
                # Send initial connection success
                yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"
                
                # Start AI planning in background thread
                def run_planning():
                    try:
                        print(f"🤖 Starting AI planner for session: {session_id}")
                        result = process_plan_for_api(username, aim, steps, event_queue)
                        
                        # Save result and competition data to database
                        conn = get_db()
                        cursor = conn.cursor()
                        
                        # Extract competition data from result
                        competition_data = result.get('competition_analysis', {})
                        
                        cursor.execute(
                            'UPDATE plans SET result = ?, competition_data = ? WHERE id = ?', 
                            (json.dumps(result), json.dumps(competition_data), plan_id)
                        )
                        conn.commit()
                        conn.close()
                        
                        # Send final result
                        event_queue.put({
                            'type': 'final_result',
                            'data': result,
                            'timestamp': datetime.now().isoformat()
                        })
                        event_queue.put(None)  # Signal completion
                        print(f"✅ Planning completed for session: {session_id}")
                        
                    except Exception as e:
                        print(f"❌ Planning error for session {session_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        event_queue.put({
                            'type': 'error',
                            'data': {'message': str(e)},
                            'timestamp': datetime.now().isoformat()
                        })
                        event_queue.put(None)
                
                planning_thread = threading.Thread(target=run_planning, daemon=True)
                planning_thread.start()
                
                # Stream events as they come
                heartbeat_counter = 0
                while True:
                    try:
                        # Wait for event with timeout
                        event = event_queue.get(timeout=30)
                        
                        if event is None:  # Completion signal
                            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                            break
                        
                        # Send event to client
                        yield f"data: {json.dumps(event)}\n\n"
                        
                    except queue.Empty:
                        # Send heartbeat to keep connection alive
                        heartbeat_counter += 1
                        yield f"data: {json.dumps({'type': 'heartbeat', 'count': heartbeat_counter})}\n\n"
                        
            except GeneratorExit:
                print(f"⚠️ Client disconnected: {session_id}")
            except Exception as e:
                print(f"❌ Stream error for {session_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Cleanup
                if session_id in active_streams:
                    del active_streams[session_id]
                print(f"🧹 Cleaned up session: {session_id}")
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        print(f"❌ Stream endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/plan', methods=['POST'])
def plan():
    """Original non-streaming endpoint"""
    try:
        data = request.json
        username = data.get('username')
        aim = data.get('aim')
        steps = data.get('steps')
        
        if not username or not aim or not steps:
            return jsonify({'success': False, 'message': 'Username, aim, and steps are required'}), 400
        
        print(f"\n{'='*60}")
        print(f"🔥 New planning request from: {username}")
        print(f"🎯 Aim: {aim}")
        print(f"📋 Steps: {steps}")
        print(f"{'='*60}\n")
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO plans (username, aim, steps) VALUES (?, ?, ?)',
                      (username, aim, json.dumps(steps)))
        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        if AI_PLANNER_AVAILABLE:
            try:
                print("🤖 Calling AI planner...")
                result = process_plan_for_api(username, aim, steps)
                
                if "error" in result:
                    print(f"❌ AI planner error: {result['error']}")
                    return jsonify({'success': False, 'message': f"AI processing failed: {result['error']}"}), 500
                
                conn = get_db()
                cursor = conn.cursor()
                
                # Extract competition data from result
                competition_data = result.get('competition_analysis', {})
                
                cursor.execute(
                    'UPDATE plans SET result = ?, competition_data = ? WHERE id = ?', 
                    (json.dumps(result), json.dumps(competition_data), plan_id)
                )
                conn.commit()
                conn.close()
                
                print(f"✅ AI planning completed successfully for {username}")
                print(f"📊 Generated {len(result)} agent plans\n")
                
                return jsonify(result), 200
                
            except Exception as e:
                print(f"❌ Error calling AI planner: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': f'AI processing error: {str(e)}'}), 500
        else:
            print("⚠️ AI planner not available")
            return jsonify({'success': False, 'message': 'AI planner not available'}), 503
        
    except Exception as e:
        print(f"❌ Planning error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/plans/<username>', methods=['GET'])
def get_plans(username):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM plans WHERE username = ? ORDER BY created_at DESC', (username,))
        plans = cursor.fetchall()
        conn.close()
        
        plans_list = []
        for plan in plans:
            plan_data = {
                'id': plan['id'],
                'aim': plan['aim'],
                'steps': json.loads(plan['steps']),
                'created_at': plan['created_at']
            }
            if plan['result']:
                plan_data['result'] = json.loads(plan['result'])
            if plan['features']:
                plan_data['features'] = json.loads(plan['features'])
            if plan['competition_data']:
                plan_data['competition_data'] = json.loads(plan['competition_data'])
            plans_list.append(plan_data)
        
        return jsonify({'success': True, 'plans': plans_list}), 200
        
    except Exception as e:
        print(f"❌ Get plans error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    try:
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != 'admin123':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, created_at FROM users ORDER BY created_at DESC')
        users = cursor.fetchall()
        conn.close()
        
        users_list = []
        for user in users:
            users_list.append({
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'created_at': user['created_at']
            })
        
        return jsonify({'success': True, 'users': users_list}), 200
        
    except Exception as e:
        print(f"❌ Get users error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/admin/plans', methods=['GET'])
def get_all_plans():
    try:
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != 'admin123':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM plans ORDER BY created_at DESC LIMIT 50')
        plans = cursor.fetchall()
        conn.close()
        
        plans_list = []
        for plan in plans:
            plan_data = {
                'id': plan['id'],
                'username': plan['username'],
                'aim': plan['aim'],
                'steps': json.loads(plan['steps']),
                'created_at': plan['created_at'],
                'has_result': plan['result'] is not None,
                'has_features': plan['features'] is not None,
                'has_competition': plan['competition_data'] is not None
            }
            plans_list.append(plan_data)
        
        return jsonify({'success': True, 'plans': plans_list}), 200
        
    except Exception as e:
        print(f"❌ Get all plans error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/admin/stats', methods=['GET'])
def get_stats():
    try:
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != 'admin123':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM users')
        total_users = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM plans')
        total_plans = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM plans WHERE result IS NOT NULL')
        completed_plans = cursor.fetchone()['count']
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_users': total_users,
                'total_plans': total_plans,
                'completed_plans': completed_plans,
                'pending_plans': total_plans - completed_plans
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Get stats error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'OK', 
        'ai_planner_available': AI_PLANNER_AVAILABLE,
        'active_streams': len(active_streams)
    }), 200

@app.route('/api/process-tool', methods=['POST'])
def process_tool():
    """Process software tool to extract tools and parameters"""
    try:
        data = request.json
        tool_name = data.get('tool_name', '').strip()
        
        if not tool_name:
            return jsonify({'success': False, 'message': 'Tool name is required'}), 400
        
        if len(tool_name) < 3:
            return jsonify({'success': False, 'message': 'Tool name must be at least 3 characters'}), 400
        
        print(f"\n{'='*70}")
        print(f"🔧 PROCESSING TOOL: {tool_name}")
        print(f"{'='*70}\n")
        
        if AI_PLANNER_AVAILABLE:
            try:
                from ai_planner import EnhancedMultiAIPlanner
                
                # Create planner instance
                planner = EnhancedMultiAIPlanner()
                
                # Process tool completely
                result = planner.process_software_tool_complete(tool_name)
                
                if result.get('success'):
                    print(f"\n✅ Tool processed successfully!")
                    print(f"📊 Results: {result['tools_found']} tools, {result['parameters_found']} parameters\n")
                    
                    return jsonify({
                        'success': True,
                        'tools_found': result['tools_found'],
                        'parameters_found': result['parameters_found'],
                        'tool_path': result['tool_path'],
                        'param_path': result['param_path']
                    }), 200
                else:
                    error_msg = result.get('error', 'Processing failed')
                    print(f"❌ Tool processing failed: {error_msg}\n")
                    return jsonify({'success': False, 'message': error_msg}), 500
                    
            except Exception as e:
                print(f"❌ Tool processing error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': str(e)}), 500
        else:
            return jsonify({'success': False, 'message': 'AI planner not available'}), 503
            
    except Exception as e:
        print(f"❌ Endpoint error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/remove-tool', methods=['POST'])
def remove_tool():
    """Remove tool and clear Excel data"""
    try:
        data = request.json
        tool_path = data.get('tool_path', r"C:\Users\LENOVO\Desktop\tool_name.xlsx")
        param_path = data.get('param_path', r"C:\Users\LENOVO\Desktop\parameter.xlsx")
        
        print(f"\n{'='*70}")
        print(f"🗑️ REMOVING TOOL DATA")
        print(f"{'='*70}")
        print(f"📂 Tool file: {tool_path}")
        print(f"📂 Parameter file: {param_path}\n")
        
        # Clear Excel files
        from openpyxl import Workbook
        
        # Clear tool file
        if os.path.exists(tool_path):
            wb = Workbook()
            ws = wb.active
            ws.title = "Tools"
            ws.append(['Tool Name', 'Category', 'Function', 'Sub-options', 'Example'])
            wb.save(tool_path)
            print(f"✅ Cleared: {tool_path}")
        else:
            # Create new file
            wb = Workbook()
            ws = wb.active
            ws.title = "Tools"
            ws.append(['Tool Name', 'Category', 'Function', 'Sub-options', 'Example'])
            wb.save(tool_path)
            print(f"✅ Created empty: {tool_path}")
        
        # Clear parameter file
        if os.path.exists(param_path):
            wb = Workbook()
            ws = wb.active
            ws.title = "Parameters"
            ws.append(['Tool Name', 'Parameter', 'Description'])
            wb.save(param_path)
            print(f"✅ Cleared: {param_path}")
        else:
            # Create new file
            wb = Workbook()
            ws = wb.active
            ws.title = "Parameters"
            ws.append(['Tool Name', 'Parameter', 'Description'])
            wb.save(param_path)
            print(f"✅ Created empty: {param_path}")
        
        print(f"\n✅ Tool data removed successfully")
        print(f"{'='*70}\n")
        
        return jsonify({'success': True, 'message': 'Tool removed and data cleared'}), 200
        
    except Exception as e:
        print(f"❌ Remove tool error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/parse-natural-language', methods=['POST'])
def parse_natural_language():
    """Convert natural language text to structured AIM + STEPS"""
    try:
        data = request.json
        user_input = data.get('input', '').strip()
        
        if not user_input:
            return jsonify({'success': False, 'message': 'Input text is required'}), 400
        
        if len(user_input) < 10:
            return jsonify({'success': False, 'message': 'Input too short. Please provide more details.'}), 400
        
        print(f"\n{'='*60}")
        print(f"🧠 Parsing natural language input...")
        print(f"📝 Input length: {len(user_input)} characters")
        print(f"{'='*60}\n")
        
        if AI_PLANNER_AVAILABLE:
            try:
                # Import the planner class
                from ai_planner import EnhancedMultiAIPlanner
                
                # Create temporary planner instance
                planner = EnhancedMultiAIPlanner()
                
                # Parse the natural language
                result = planner.parse_natural_language_to_structure(user_input)
                
                if "error" in result:
                    print(f"❌ Parsing failed: {result['error']}")
                    return jsonify({'success': False, 'message': result['error']}), 500
                
                print(f"✅ Successfully parsed!")
                print(f"🎯 AIM: {result['aim']}")
                print(f"📋 STEPS: {len(result['steps'])} extracted\n")
                
                return jsonify({
                    'success': True,
                    'aim': result['aim'],
                    'steps': result['steps'],
                    'raw_response': result.get('raw_response', '')
                }), 200
                
            except Exception as e:
                print(f"❌ Error in natural language parsing: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': f'Parsing error: {str(e)}'}), 500
        else:
            print("⚠️ AI planner not available")
            return jsonify({'success': False, 'message': 'AI planner not available'}), 503
        
    except Exception as e:
        print(f"❌ Endpoint error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/competition/<int:plan_id>', methods=['GET'])
def get_competition_data(plan_id):
    """Get competition analysis for a specific plan"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT competition_data, aim, username FROM plans WHERE id = ?', (plan_id,))
        plan = cursor.fetchone()
        conn.close()
        
        if not plan:
            return jsonify({'success': False, 'message': 'Plan not found'}), 404
        
        if not plan['competition_data']:
            return jsonify({'success': False, 'message': 'No competition data available'}), 404
        
        competition_data = json.loads(plan['competition_data'])
        
        return jsonify({
            'success': True,
            'plan_id': plan_id,
            'aim': plan['aim'],
            'username': plan['username'],
            'competition': competition_data
        }), 200
        
    except Exception as e:
        print(f"❌ Get competition error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.route('/api/admin/competition-stats', methods=['GET'])
def get_competition_stats():
    """Get aggregate competition statistics (admin only)"""
    try:
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != 'admin123':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Plans with competition data
        cursor.execute('SELECT COUNT(*) as count FROM plans WHERE competition_data IS NOT NULL')
        plans_with_competition = cursor.fetchone()['count']
        
        # Total competitors found (aggregate)
        cursor.execute('SELECT competition_data FROM plans WHERE competition_data IS NOT NULL')
        all_competition_data = cursor.fetchall()
        
        total_aim_competitors = 0
        total_step_competitors = 0
        total_clusters = 0
        
        for row in all_competition_data:
            try:
                comp_data = json.loads(row['competition_data'])
                
                # Count AIM competitors
                if comp_data.get('aim_level'):
                    total_aim_competitors += len(comp_data['aim_level'].get('aim_competitors', []))
                
                # Count step competitors
                if comp_data.get('step_level'):
                    for step_analysis in comp_data['step_level']:
                        total_step_competitors += step_analysis.get('total_companies', 0)
                
                # Count clusters
                if comp_data.get('summary_level'):
                    total_clusters += len(comp_data['summary_level'].get('cluster_analysis', []))
                    
            except:
                continue
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'plans_with_competition': plans_with_competition,
                'total_aim_competitors': total_aim_competitors,
                'total_step_competitors': total_step_competitors,
                'total_clusters_analyzed': total_clusters,
                'avg_aim_competitors_per_plan': round(total_aim_competitors / max(plans_with_competition, 1), 2),
                'avg_step_competitors_per_plan': round(total_step_competitors / max(plans_with_competition, 1), 2)
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Get competition stats error: {str(e)}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    init_db()
    print("\n" + "="*60)
    print("🚀 AI PLANNER BACKEND SERVER WITH FEATURE DETECTION")
    print("="*60)
    print("📊 Database: planner.db")
    print(f"🤖 AI Planner: {'✅ Available' if AI_PLANNER_AVAILABLE else '⚠️ Not available'}")
    print("🌐 Server: http://localhost:5000")
    print("🔴 Streaming: /api/plan/stream (SSE)")
    print("📡 Standard: /api/plan (JSON)")
    print("✨ Features: Real-time detection enabled")
    print("🏢 Competition: Real-time analysis enabled")
    print("⏱️  Request timeout: 5 hours")
    print("="*60)
    print("\n📍 API Endpoints:")
    print("   - POST /api/plan/stream       → Real-time planning with SSE")
    print("   - POST /api/parse-natural-language → Convert text to AIM+STEPS")
    print("   - GET  /api/competition/<id>  → Get competition data for plan")
    print("   - GET  /api/admin/competition-stats → Admin competition statistics")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
