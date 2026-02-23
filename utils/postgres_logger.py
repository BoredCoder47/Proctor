import psycopg2
import os
import time
from dotenv import load_dotenv


class PostgresLogger:
    def __init__(self):
        load_dotenv()
        self.database_url = os.getenv("DATABASE_URL")

        if not self.database_url:
            raise EnvironmentError("DATABASE_URL not found.")

        self._connect()

    # ---------------- CONNECT ----------------
    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                self.database_url,
                sslmode="require",
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            self.cur = self.conn.cursor()
            self._ensure_tables()
            print("✅ Postgres connected")

        except Exception as e:
            print("❌ DB connect failed:", e)
            raise

    def _safe_execute(self, query, params):
        try:
            self.cur.execute(query, params)
            self.conn.commit()
        except Exception:
            print("⚠️ DB reconnecting...")
            self._connect()
            self.cur.execute(query, params)
            self.conn.commit()

    # ---------------- TABLES ----------------
    def _ensure_tables(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                frame_id INTEGER,
                timestamp TIMESTAMP,
                recognized_name TEXT,
                expected_user TEXT,
                face_visible BOOLEAN,
                eyes_visible BOOLEAN,
                looking_away BOOLEAN,
                multiple_faces BOOLEAN,
                imposter_detected BOOLEAN,
                frame_url TEXT
            );
        """)

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_anomalies (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                timestamp TIMESTAMP,
                noise_detected BOOLEAN,
                energy FLOAT
            );
        """)

        self.conn.commit()

    # ---------------- VIDEO ----------------
    def log_anomaly(
        self,
        frame_id,
        name,
        expected_user,
        face_visible,
        eyes_visible,
        looking_away,
        multiple_faces,
        imposter_detected,
        session_id,
        frame_url=None
    ):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        self._safe_execute("""
            INSERT INTO anomalies (
                session_id, frame_id, timestamp,
                recognized_name, expected_user,
                face_visible, eyes_visible, looking_away,
                multiple_faces, imposter_detected, frame_url
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            session_id,
            frame_id,
            ts,
            name,
            expected_user,
            face_visible,
            eyes_visible,
            looking_away,
            multiple_faces,
            imposter_detected,
            frame_url
        ))

    # ---------------- AUDIO ----------------
    def log_audio(self, session_id, noise, energy):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        self._safe_execute("""
            INSERT INTO audio_anomalies (
                session_id, timestamp, noise_detected, energy
            )
            VALUES (%s,%s,%s,%s)
        """, (session_id, ts, noise, energy))

    # ---------------- CLEANUP ----------------
    def close(self):
        try:
            self.cur.close()
            self.conn.close()
        except:
            pass