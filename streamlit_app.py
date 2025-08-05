#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
import streamlit as st
import google.generativeai as genai
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, func, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
WATCH_DIR  = os.getenv("WATCH_DIRECTORY", "")
DB_URL     = os.getenv("DATABASE_URL", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Database setup ───────────────────────────────────────────────────────────
Base    = declarative_base()
engine  = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

# Inspect existing schema to handle migrations
insp    = inspect(engine)
has_log = insp.has_table("log_entries")
cols    = [c["name"] for c in insp.get_columns("log_entries")] if has_log else []

class LogEntry(Base):
    __tablename__ = 'log_entries'
    id               = Column(Integer, primary_key=True, autoincrement=True)
    date             = Column(String)
    time             = Column(String)
    policy_identity  = Column(String)
    internal_ip      = Column(String)
    external_ip      = Column(String)
    action           = Column(String)
    destination      = Column(String)
    categories       = Column(String)
    if 'source_file' in cols:
        source_file = Column(String)

# Auto-add missing columns
if has_log:
    # List of columns we expect to have
    expected_columns = [
        'date', 'time', 'policy_identity', 'internal_ip', 
        'external_ip', 'action', 'destination', 'categories', 'source_file'
    ]
    
    # Check for missing columns and add them
    with engine.connect() as conn:
        for col in expected_columns:
            if col not in cols and col != 'id':  # id is always present
                conn.execute(text(f"ALTER TABLE log_entries ADD COLUMN {col} VARCHAR;"))

class ProcessedFile(Base):
    __tablename__ = 'processed_files'
    filename     = Column(String, primary_key=True)
    processed_at = Column(DateTime, default=func.now())

Base.metadata.create_all(engine)

# ── CSV ingestion ────────────────────────────────────────────────────────────
class CSVHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.csv'):
            ingest_csv(event.src_path)

def start_watcher():
    os.makedirs(WATCH_DIR, exist_ok=True)
    session   = Session()
    processed = {pf.filename for pf in session.query(ProcessedFile).all()}
    session.close()
    for fname in os.listdir(WATCH_DIR):
        path = os.path.join(WATCH_DIR, fname)
        if fname.lower().endswith('.csv') and fname not in processed:
            ingest_csv(path)
    obs = Observer()
    obs.schedule(CSVHandler(), WATCH_DIR, recursive=False)
    obs.start()
    return obs

def ingest_csv(path):
    fname   = os.path.basename(path)
    session = Session()
    try:
        if session.query(ProcessedFile).filter_by(filename=fname).first():
            return
        df = pd.read_csv(path, sep=';')
        for _, row in df.iterrows():
            data = {
                'date':            row['Date'],
                'time':            row['Time'],
                'policy_identity': row['Policy Identity'],
                'internal_ip':     row['Internal IP Address'],
                'external_ip':     row['External IP Address'],
                'action':          row['Action'],
                'destination':     row['Destination'],
                'categories':      row['Categories']
            }
            if 'source_file' in cols:
                data['source_file'] = fname
            session.add(LogEntry(**data))
        session.add(ProcessedFile(filename=fname))
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Failed ingest {fname}: {e}")
    finally:
        session.close()

# ── Tool Implementations ─────────────────────────────────────────────────────
def list_tools(_):
    """Return plain text description of tools"""
    tool_list = []
    for name, spec in TOOL_DEFINITIONS.items():
        if name == "none": continue
        args = ', '.join(spec['required_args']) if spec['required_args'] else 'None'
        tool_list.append(f"- {name}: {spec['description']} (Required args: {args})")
    return "Available Tools:\n" + "\n".join(tool_list)

def count_lines(_):
    session = Session()
    count = session.query(LogEntry).count()
    session.close()
    return count

def get_line_by_id(args):
    session = Session()
    entry = session.get(LogEntry, int(args['id']))
    session.close()
    if not entry:
        return "No entry found with that ID"
    return {c.name: getattr(entry, c.name) for c in LogEntry.__table__.c}

def _parse_datetime(dt_str):
    """Parse combined date/time strings into separate components"""
    # Try different formats
    formats = [
        "%d/%m/%Y %H:%M",   # 27/07/2025 13:30
        "%d/%m/%Y %H",       # 27/07/2025 13
        "%d/%m/%Y",          # 27/07/2025
        "%d/%m/%y %H:%M",    # 27/07/25 13:30
        "%d/%m/%y %H",       # 27/07/25 13
        "%d/%m/%y"           # 27/07/25
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M")
        except ValueError:
            continue
    return None, None

def _query_entries(session, args):
    """Generic query builder for date/time filters"""
    query = session.query(LogEntry)
    
    # Handle combined date/time queries
    if 'datetime' in args:
        date_part, time_part = _parse_datetime(args['datetime'])
        if date_part:
            query = query.filter(LogEntry.date.ilike(f"%{date_part}%"))
        if time_part:
            # Remove seconds if present in time_part
            time_part = time_part.split(':')[0] + ':%' if ':' in time_part else time_part + ':%'
            query = query.filter(LogEntry.time.ilike(f"%{time_part}%"))
    
    # Handle separate date/time queries
    if 'date' in args:
        query = query.filter(LogEntry.date.ilike(f"%{args['date']}%"))
    if 'time' in args:
        # Handle partial time formats (like "13" or "13:30")
        time_val = args['time']
        if ':' not in time_val:
            time_val += ':%'  # Search for hour:minute
        query = query.filter(LogEntry.time.ilike(f"%{time_val}%"))
    
    # Handle other filters
    for col in ['policy_identity', 'internal_ip', 'external_ip', 'action', 'destination']:
        if col in args:
            col_attr = getattr(LogEntry, col)
            query = query.filter(col_attr.ilike(f"%{args[col]}%"))
    
    return query

def count_entries(args):
    session = Session()
    try:
        query = _query_entries(session, args)
        count = query.count()
        return count
    finally:
        session.close()

def get_entries(args):
    session = Session()
    try:
        query = _query_entries(session, args)
        entries = query.all()
        return [{
            'id': e.id,
            'date': e.date,
            'time': e.time,
            'policy_identity': e.policy_identity,
            'internal_ip': e.internal_ip,
            'external_ip': e.external_ip,
            'action': e.action,
            'destination': e.destination,
            'categories': e.categories
        } for e in entries]
    finally:
        session.close()

# ── Tool Registry ────────────────────────────────────────────────────────────
TOOL_DEFINITIONS = {
    'list_tools': {
        'description': 'List all available tools',
        'required_args': [],
        'func': list_tools
    },
    'count_lines': {
        'description': 'Count all log entries',
        'required_args': [],
        'func': count_lines
    },
    'count_entries': {
        'description': 'Count entries by date/time or other filters',
        'required_args': [],
        'func': count_entries
    },
    'get_line_by_id': {
        'description': 'Get log entry by ID',
        'required_args': ['id'],
        'func': get_line_by_id
    },
    'get_entries': {
        'description': 'Get entries by date/time or other filters',
        'required_args': [],
        'func': get_entries
    },
    'none': {
        'description': 'No tool needed - normal conversation',
        'required_args': [],
        'func': lambda _: "No tool executed"
    }
}

# ── Streamlit + Gemini Chatbot ───────────────────────────────────────────────
def setup_gemini():
    if not GEMINI_KEY:
        st.error("Missing GEMINI_API_KEY")
        st.stop()
    genai.configure(api_key=GEMINI_KEY)
    return genai.GenerativeModel('gemini-2.0-flash')

st.set_page_config(page_title="Log Analytics Chatbot", layout="wide")
st.title("🤖 Log Analytics Chatbot")

# Start the watcher
if 'watcher' not in st.session_state:
    st.session_state.watcher = start_watcher()
# Initialize history
if 'messages' not in st.session_state:
    st.session_state.messages = []

model = setup_gemini()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Handle new user input
if prompt := st.chat_input("Ask your question…"):
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Build tool invocation prompt
    tool_list_str = "\n".join(
        f"- {name}: {spec['description']} (Args: {', '.join(spec['required_args']) or 'None'})"
        for name, spec in TOOL_DEFINITIONS.items() if name != "none"
    )
    system_prompt = (
        "You are an AI assistant with access to these tools:\n"
        f"{tool_list_str}\n\n"
        "Decide whether to use a tool or respond normally:\n"
        "- If using a tool, reply ONLY with JSON: {'name':'<tool_name>','arguments':{...}}\n"
        "- If not using a tool, respond naturally\n"
        "Available filter arguments for count_entries and get_entries:\n"
        "  - 'date': 'DD/MM/YYYY' (e.g., '29/07/2025')\n"
        "  - 'time': 'HH:MM' or 'HH' (e.g., '13:30' or '13')\n"
        "  - 'datetime': 'DD/MM/YYYY HH:MM' (combined format)\n"
        "  - 'internal_ip': IP address (e.g., '107.78.99.191')\n"
        "  - 'external_ip': IP address\n"
        "  - 'action': action type (e.g., 'Blocked', 'Allowed')\n"
        "  - 'destination': destination URL/domain\n"
        "  - 'policy_identity': policy name\n"
        "You can combine multiple filters in one query!\n\n"
        "Examples:\n"
        "  User: How many entries from 29/07/2025?\n"
        "  Assistant: {\"name\":\"count_entries\",\"arguments\":{\"date\":\"29/07/2025\"}}\n"
        "  User: Get me lines with internal ip 107.78.99.191 and happened at 29/07/2025 13:13\n"
        "  Assistant: {\"name\":\"get_entries\",\"arguments\":{\"internal_ip\":\"107.78.99.191\",\"date\":\"29/07/2025\",\"time\":\"13:13\"}}\n"
        "  User: Count blocked entries from internal IP 192.168.1.1 on 30/07/2025\n"
        "  Assistant: {\"name\":\"count_entries\",\"arguments\":{\"action\":\"Blocked\",\"internal_ip\":\"192.168.1.1\",\"date\":\"30/07/2025\"}}\n"
        "  User: Show me entries for mail.example.com that were blocked\n"
        "  Assistant: {\"name\":\"get_entries\",\"arguments\":{\"destination\":\"mail.example.com\",\"action\":\"Blocked\"}}\n"
        "  User: What tools do you have?\n"
        "  Assistant: {\"name\":\"list_tools\",\"arguments\":{}}"
    )

    try:
        resp = model.generate_content(system_prompt + f"\nUser: {prompt}")
        raw  = resp.text.strip()

        # Try to parse as JSON tool call
        try:
            call = json.loads(raw)
            name = call.get("name", "none")
            args = call.get("arguments", {})
            
            # Execute the tool
            if name in TOOL_DEFINITIONS:
                spec = TOOL_DEFINITIONS[name]
                # Validate required arguments
                missing = [r for r in spec['required_args'] if r not in args]
                if missing:
                    err = f"Missing argument(s): {', '.join(missing)}"
                    st.session_state.messages.append({'role':'assistant','content':err})
                    with st.chat_message('assistant'):
                        st.error(err)
                else:
                    result = spec['func'](args)
                    
                    # Format the result based on tool type
                    if name in ['get_entries', 'get_line_by_id']:
                        # JSON format for data retrieval tools
                        if isinstance(result, str) and result == "No entry found with that ID":
                            out = "No entries found"
                        elif isinstance(result, list) and len(result) == 0:
                            out = "No entries found"
                        else:
                            if name == 'get_entries':
                                out = f"📋 JSON Results ({len(result)} entries):\n```json\n{json.dumps(result, indent=2)}\n```"
                            else:  # get_line_by_id
                                out = f"📄 JSON Entry Details:\n```json\n{json.dumps(result, indent=2)}\n```"
                    else:
                        # Human language format for other tools
                        if isinstance(result, int):
                            out = f"🔢 Result: {result}"
                        elif isinstance(result, str):
                            out = result
                        else:
                            out = str(result)
                        
                    st.session_state.messages.append({'role':'assistant','content':out})
                    with st.chat_message('assistant'):
                        st.markdown(out)
            else:
                # Handle invalid tool call
                st.session_state.messages.append({'role':'assistant','content':raw})
                with st.chat_message('assistant'):
                    st.markdown(raw)
        
        except json.JSONDecodeError:
            # Natural language response
            st.session_state.messages.append({'role':'assistant','content':raw})
            with st.chat_message('assistant'):
                st.markdown(raw)
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        st.session_state.messages.append({'role':'assistant','content':error_msg})
        with st.chat_message('assistant'):
            st.error(error_msg)

# Quit button
if st.button("Quit"):
    st.session_state.watcher.stop()
    st.session_state.clear()
    st.experimental_rerun()
