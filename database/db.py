import sqlite3
import json
import os
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sems.db"
)


def now_str():
    return datetime.now().isoformat(timespec="seconds")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _column_exists(cur, table, column):
    return any(
        r[1] == column
        for r in cur.execute(f"PRAGMA table_info({table})")
    )


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            faculty TEXT NOT NULL,
            email TEXT,
            descriptor TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(name, faculty)
        )
    """)

    # Upgrade older databases that were created before the email column.
    if not _column_exists(cur, "students", "email"):
        cur.execute("ALTER TABLE students ADD COLUMN email TEXT")

    try:
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_students_email "
            "ON students(email) WHERE email IS NOT NULL"
        )
    except sqlite3.OperationalError:
        # Old data already contains duplicate emails; skip the index.
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            code TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'available',
            added_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)

    # Seed one default admin account if none exists yet.
    existing = cur.execute(
        "SELECT COUNT(*) FROM admins"
    ).fetchone()[0]

    if existing == 0:
        cur.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("admin123")),
        )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            equipment_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            requested_at TEXT NOT NULL,
            decided_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(equipment_id) REFERENCES equipment(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS returns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL UNIQUE,
            returned_at TEXT NOT NULL,
            FOREIGN KEY(request_id) REFERENCES requests(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            item_name TEXT NOT NULL,
            notes TEXT,
            requested_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # Charges the admin sets on a borrow record.
    # Students pay by scanning the admin's payment QR.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            request_id INTEGER,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'due',
            note TEXT,
            reference TEXT,
            payment_proof TEXT,
            created_at TEXT NOT NULL,
            submitted_at TEXT,
            paid_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(request_id) REFERENCES requests(id)
        )
    """)

    # Upgrade older payments table.
    if not _column_exists(cur, "payments", "request_id"):
        cur.execute(
            "ALTER TABLE payments ADD COLUMN request_id INTEGER"
        )

    if not _column_exists(cur, "payments", "note"):
        cur.execute(
            "ALTER TABLE payments ADD COLUMN note TEXT"
        )

    # New column for student's payment screenshot.
    if not _column_exists(cur, "payments", "payment_proof"):
        cur.execute(
            "ALTER TABLE payments ADD COLUMN payment_proof TEXT"
        )

    has_reports_table = cur.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='broken_reports'"
    ).fetchone()

    if has_reports_table and _column_exists(
        cur, "payments", "report_id"
    ):
        cur.execute("""
            UPDATE payments
            SET request_id = (
                    SELECT br.request_id
                    FROM broken_reports br
                    WHERE br.id = payments.report_id
                ),
                note = COALESCE(
                    note,
                    (
                        SELECT br.admin_note
                        FROM broken_reports br
                        WHERE br.id = payments.report_id
                    )
                )
            WHERE request_id IS NULL
              AND report_id IS NOT NULL
        """)

    # Simple key/value settings.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


# ---------------- Settings ----------------

def get_setting(key, default=None):
    conn = get_db()

    row = conn.execute(
        "SELECT value FROM settings WHERE key = ?",
        (key,)
    ).fetchone()

    conn.close()

    return (
        row["value"]
        if row and row["value"] is not None
        else default
    )


def set_setting(key, value):
    conn = get_db()

    if value is None:
        conn.execute(
            "DELETE FROM settings WHERE key = ?",
            (key,)
        )
    else:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    conn.commit()
    conn.close()


# ---------------- Admin ----------------

def get_admin_by_username(username):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM admins WHERE username = ?",
        (username.strip(),)
    ).fetchone()

    conn.close()

    return row


# ---------------- Students ----------------

def add_student(name, faculty, email, descriptor):
    conn = get_db()

    try:
        conn.execute(
            "INSERT INTO students "
            "(name, faculty, email, descriptor, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                name.strip(),
                faculty.strip(),
                (email or "").strip().lower() or None,
                json.dumps(descriptor),
                now_str(),
            ),
        )

        conn.commit()

    finally:
        conn.close()


def get_student_by_name_faculty(name, faculty):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM students "
        "WHERE name = ? AND faculty = ?",
        (name.strip(), faculty.strip()),
    ).fetchone()

    conn.close()

    return row


def get_student_by_email(email):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM students WHERE email = ?",
        ((email or "").strip().lower(),)
    ).fetchone()

    conn.close()

    return row


def get_student_by_id(student_id):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()

    conn.close()

    return row


def get_all_students():
    conn = get_db()

    rows = conn.execute(
        "SELECT * FROM students ORDER BY created_at DESC"
    ).fetchall()

    conn.close()

    return rows


def delete_student(student_id):
    """Raises sqlite3.IntegrityError if the student has borrow/report history."""
    conn = get_db()

    try:
        conn.execute(
            "DELETE FROM wishlist WHERE student_id = ?",
            (student_id,)
        )

        conn.execute(
            "DELETE FROM students WHERE id = ?",
            (student_id,)
        )

        conn.commit()

    except sqlite3.IntegrityError:
        conn.rollback()
        raise

    finally:
        conn.close()


# ---------------- Equipment ----------------

def add_equipment(name, category):
    code = uuid.uuid4().hex[:8].upper()

    conn = get_db()

    conn.execute(
        "INSERT INTO equipment "
        "(name, category, code, status, added_at) "
        "VALUES (?, ?, ?, 'available', ?)",
        (
            name.strip(),
            category.strip(),
            code,
            now_str()
        ),
    )

    conn.commit()
    conn.close()

    return code


def get_all_equipment():
    conn = get_db()

    rows = conn.execute(
        "SELECT * FROM equipment ORDER BY added_at DESC"
    ).fetchall()

    conn.close()

    return rows


def get_equipment_by_code(code):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM equipment WHERE code = ?",
        (code.strip(),)
    ).fetchone()

    conn.close()

    return row


def get_equipment_by_id(equipment_id):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM equipment WHERE id = ?",
        (equipment_id,)
    ).fetchone()

    conn.close()

    return row


def update_equipment_status(equipment_id, status):
    conn = get_db()

    conn.execute(
        "UPDATE equipment SET status = ? WHERE id = ?",
        (status, equipment_id)
    )

    conn.commit()
    conn.close()


def delete_equipment(equipment_id):
    """Raises sqlite3.IntegrityError if the equipment has borrow/report history."""
    conn = get_db()

    try:
        conn.execute(
            "DELETE FROM equipment WHERE id = ?",
            (equipment_id,)
        )

        conn.commit()

    except sqlite3.IntegrityError:
        conn.rollback()
        raise

    finally:
        conn.close()


# ---------------- Borrow records ----------------

def record_borrow(student_id, equipment_id):
    """Instantly records a borrow."""
    conn = get_db()

    now = now_str()

    conn.execute(
        "INSERT INTO requests "
        "(student_id, equipment_id, status, requested_at, decided_at) "
        "VALUES (?, ?, 'approved', ?, ?)",
        (student_id, equipment_id, now, now),
    )

    conn.execute(
        "UPDATE equipment SET status = 'borrowed' "
        "WHERE id = ?",
        (equipment_id,)
    )

    conn.commit()
    conn.close()


def get_active_request_for_equipment(equipment_id):
    """The approved currently borrowed request."""
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM requests "
        "WHERE equipment_id = ? AND status = 'approved'",
        (equipment_id,),
    ).fetchone()

    conn.close()

    return row


def get_request_by_id(request_id):
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM requests WHERE id = ?",
        (request_id,)
    ).fetchone()

    conn.close()

    return row


def get_borrowed_by_student(student_id):
    """Equipment currently borrowed by this student."""
    conn = get_db()

    rows = conn.execute(
        """
        SELECT e.*, 
               r.id AS request_id,
               r.decided_at AS borrowed_at
        FROM requests r
        JOIN equipment e
            ON e.id = r.equipment_id
        WHERE r.student_id = ?
          AND r.status = 'approved'
        ORDER BY r.decided_at DESC
        """,
        (student_id,),
    ).fetchall()

    conn.close()

    return rows


def get_history_by_student(student_id, limit=20):
    conn = get_db()

    rows = conn.execute(
        """
        SELECT r.*,
               e.name AS equipment_name,
               e.category AS category,
               ret.returned_at AS returned_at
        FROM requests r
        JOIN equipment e
            ON e.id = r.equipment_id
        LEFT JOIN returns ret
            ON ret.request_id = r.id
        WHERE r.student_id = ?
          AND r.status IN ('returned', 'broken')
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (student_id, limit),
    ).fetchall()

    conn.close()

    return rows


# ---------------- Returns ----------------

def add_return(request_id):
    conn = get_db()

    req = conn.execute(
        "SELECT * FROM requests WHERE id = ?",
        (request_id,)
    ).fetchone()

    if req:
        now = now_str()

        conn.execute(
            "INSERT INTO returns "
            "(request_id, returned_at) VALUES (?, ?)",
            (request_id, now),
        )

        conn.execute(
            "UPDATE requests SET status = 'returned' "
            "WHERE id = ?",
            (request_id,)
        )

        conn.execute(
            "UPDATE equipment SET status = 'available' "
            "WHERE id = ?",
            (req["equipment_id"],)
        )

        conn.commit()

    conn.close()


# ---------------- Wishlist ----------------

def add_wishlist_request(student_id, item_name, notes):
    conn = get_db()

    conn.execute(
        "INSERT INTO wishlist "
        "(student_id, item_name, notes, requested_at) "
        "VALUES (?, ?, ?, ?)",
        (
            student_id,
            item_name.strip(),
            (notes or "").strip(),
            now_str()
        ),
    )

    conn.commit()
    conn.close()


def get_all_wishlist_requests():
    conn = get_db()

    rows = conn.execute(
        """
        SELECT w.*,
               s.name AS student_name,
               s.faculty AS faculty
        FROM wishlist w
        JOIN students s
            ON s.id = w.student_id
        ORDER BY w.requested_at DESC
        """
    ).fetchall()

    conn.close()

    return rows


def delete_wishlist_request(wishlist_id):
    conn = get_db()

    conn.execute(
        "DELETE FROM wishlist WHERE id = ?",
        (wishlist_id,)
    )

    conn.commit()
    conn.close()


# ---------------- Student: borrow records with pay status ----------------

def get_records_by_student(student_id):
    """
    Every borrow of this student:
    borrowed time, return time and charge/pay status.
    """

    conn = get_db()

    rows = conn.execute(
        """
        SELECT r.id AS request_id,
               r.status AS record_status,
               r.requested_at AS borrowed_at,

               e.name AS equipment_name,
               e.category AS category,
               e.code AS code,

               ret.returned_at AS returned_at,

               p.id AS payment_id,
               p.amount AS amount,
               p.status AS payment_status,
               p.note AS payment_note,
               p.reference AS reference,

               p.payment_proof AS payment_proof,

               p.paid_at AS paid_at

        FROM requests r

        JOIN equipment e
            ON e.id = r.equipment_id

        LEFT JOIN returns ret
            ON ret.request_id = r.id

        LEFT JOIN payments p
            ON p.request_id = r.id

        WHERE r.student_id = ?

        ORDER BY
            CASE r.status
                WHEN 'approved' THEN 0
                ELSE 1
            END,
            r.id DESC
        """,
        (student_id,),
    ).fetchall()

    conn.close()

    return rows


def count_unpaid_for_student(student_id):
    conn = get_db()

    n = conn.execute(
        "SELECT COUNT(*) FROM payments "
        "WHERE student_id = ? AND status = 'due'",
        (student_id,),
    ).fetchone()[0]

    conn.close()

    return n


# ---------------- Payments ----------------

_PAYMENT_SELECT = """
    SELECT p.*,

           s.name AS student_name,
           s.faculty AS faculty,

           e.name AS equipment_name,
           e.category AS category,

           r.requested_at AS borrowed_at,

           ret.returned_at AS returned_at

    FROM payments p

    JOIN students s
        ON s.id = p.student_id

    LEFT JOIN requests r
        ON r.id = p.request_id

    LEFT JOIN equipment e
        ON e.id = r.equipment_id

    LEFT JOIN returns ret
        ON ret.request_id = r.id
"""


def get_all_payments():
    conn = get_db()

    rows = conn.execute(
        _PAYMENT_SELECT
        + """
        ORDER BY
            CASE p.status
                WHEN 'submitted' THEN 0
                WHEN 'due' THEN 1
                ELSE 2
            END,
            p.id DESC
        """
    ).fetchall()

    conn.close()

    return rows


def get_requests_without_charge(limit=200):
    """Borrow records the admin can still put a charge on."""
    conn = get_db()

    rows = conn.execute(
        """
        SELECT r.id AS request_id,
               r.requested_at AS borrowed_at,
               r.status AS record_status,

               s.name AS student_name,
               s.faculty AS faculty,

               e.name AS equipment_name

        FROM requests r

        JOIN students s
            ON s.id = r.student_id

        JOIN equipment e
            ON e.id = r.equipment_id

        LEFT JOIN payments p
            ON p.request_id = r.id

        WHERE p.id IS NULL

        ORDER BY r.id DESC

        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    conn.close()

    return rows


def add_charge(request_id, amount, note):
    """
    Puts a charge on a borrow record.
    Returns:
    "ok", "exists", or "missing".
    """

    conn = get_db()

    req = conn.execute(
        "SELECT * FROM requests WHERE id = ?",
        (request_id,)
    ).fetchone()

    if not req:
        conn.close()
        return "missing"

    if conn.execute(
        "SELECT 1 FROM payments WHERE request_id = ?",
        (request_id,)
    ).fetchone():

        conn.close()
        return "exists"

    conn.execute(
        """
        INSERT INTO payments
            (
                student_id,
                request_id,
                amount,
                status,
                note,
                created_at
            )
        VALUES
            (?, ?, ?, 'due', ?, ?)
        """,
        (
            req["student_id"],
            request_id,
            amount,
            (note or "").strip() or None,
            now_str(),
        ),
    )

    conn.commit()
    conn.close()

    return "ok"


def student_mark_payment_submitted(
    payment_id,
    student_id,
    reference,
    payment_proof
):
    """
    Student says 'I have paid'.

    The payment screenshot filename is stored so
    the admin can view the submitted proof.
    """

    conn = get_db()

    cur = conn.execute(
        """
        UPDATE payments

        SET status = 'submitted',
            reference = ?,
            payment_proof = ?,
            submitted_at = ?

        WHERE id = ?
          AND student_id = ?
          AND status = 'due'
        """,
        (
            (reference or "").strip() or None,
            payment_proof,
            now_str(),
            payment_id,
            student_id,
        ),
    )

    conn.commit()

    changed = cur.rowcount > 0

    conn.close()

    return changed


def admin_confirm_payment(payment_id):
    conn = get_db()

    cur = conn.execute(
        """
        UPDATE payments
        SET status = 'paid',
            paid_at = ?
        WHERE id = ?
          AND status != 'paid'
        """,
        (
            now_str(),
            payment_id
        ),
    )

    conn.commit()

    changed = cur.rowcount > 0

    conn.close()

    return changed


def admin_cancel_payment(payment_id):
    """Removes a charge that has not been paid yet."""

    conn = get_db()

    cur = conn.execute(
        """
        DELETE FROM payments
        WHERE id = ?
          AND status != 'paid'
        """,
        (payment_id,)
    )

    conn.commit()

    changed = cur.rowcount > 0

    conn.close()

    return changed


def count_pending_payments():
    conn = get_db()

    n = conn.execute(
        "SELECT COUNT(*) FROM payments "
        "WHERE status != 'paid'"
    ).fetchone()[0]

    conn.close()

    return n


# ---------------- Reports (admin) ----------------

def get_all_requests_report():
    conn = get_db()

    rows = conn.execute(
        """
        SELECT r.*,

               s.name AS student_name,
               s.faculty AS faculty,

               e.name AS equipment_name,
               e.category AS category,
               e.code AS code,

               ret.returned_at AS returned_at

        FROM requests r

        JOIN students s
            ON s.id = r.student_id

        JOIN equipment e
            ON e.id = r.equipment_id

        LEFT JOIN returns ret
            ON ret.request_id = r.id

        ORDER BY r.id DESC
        """
    ).fetchall()

    conn.close()

    return rows


def get_report_summary():
    conn = get_db()

    one = lambda q: conn.execute(q).fetchone()[0]

    summary = {
        "total_requests": one(
            "SELECT COUNT(*) FROM requests"
        ),

        "approved": one(
            "SELECT COUNT(*) FROM requests "
            "WHERE status = 'approved'"
        ),

        "returned": one(
            "SELECT COUNT(*) FROM requests "
            "WHERE status = 'returned'"
        ),

        "broken": one(
            "SELECT COUNT(*) FROM equipment "
            "WHERE status = 'broken'"
        ),

        "wishlist_count": one(
            "SELECT COUNT(*) FROM wishlist"
        ),

        "collected": one(
            "SELECT COALESCE(SUM(amount), 0) "
            "FROM payments WHERE status = 'paid'"
        ),

        "outstanding": one(
            "SELECT COALESCE(SUM(amount), 0) "
            "FROM payments WHERE status != 'paid'"
        ),
    }

    conn.close()

    return summary