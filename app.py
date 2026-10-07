from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash
)
from werkzeug.security import check_password_hash
from database import db
from face_engine import recognition, liveness
from face_engine.recognition import FaceEngineUnavailable

import base64
import json
import numpy as np
import cv2
import os
import re
import time
import sqlite3
import uuid


app = Flask(__name__)
app.secret_key = "sems-secret-key-change-me"

# Currency label shown next to payment amounts.
CURRENCY = "Rs."


# ---------------------------------------------------------------------
# Upload folders
# ---------------------------------------------------------------------

# Payment QR uploads are saved here.
UPLOAD_DIR = os.path.join(
    app.static_folder,
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# Student payment screenshots are saved here.
PAYMENT_PROOF_DIR = os.path.join(
    UPLOAD_DIR,
    "payment_proofs"
)

os.makedirs(
    PAYMENT_PROOF_DIR,
    exist_ok=True
)


ALLOWED_QR_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif"
}


# Reject uploads bigger than 4 MB
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

app.jinja_env.globals["currency"] = CURRENCY


with app.app_context():
    db.init_db()


# ---------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------

def decode_base64_image(data_url):
    """
    Turns a data:image/jpeg;base64,... string
    from the browser into an OpenCV BGR image.
    """
    try:
        encoded = (
            data_url.split(",", 1)[1]
            if "," in data_url
            else data_url
        )

        binary = base64.b64decode(encoded)

        arr = np.frombuffer(
            binary,
            dtype=np.uint8
        )

        return cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR
        )

    except Exception:
        return None


# ---------------------------------------------------------------------
# Sports icons
# ---------------------------------------------------------------------

SPORT_ICONS = {
    "football": "⚽",
    "soccer": "⚽",
    "badminton": "🏸",
    "basketball": "🏀",
    "volleyball": "🏐",
    "table tennis": "🏓",
    "ping pong": "🏓",
    "cricket": "🏏",
    "hockey": "🏑",
    "tennis": "🎾",
    "chess": "♟️",
    "rugby": "🏉",
    "baseball": "⚾",
}


def sport_icon(category):
    return SPORT_ICONS.get(
        (category or "").strip().lower(),
        "🏅"
    )


app.jinja_env.globals["sport_icon"] = sport_icon


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def student_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):

        if not session.get("student_id"):
            return redirect(
                url_for("student_login")
            )

        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):

        if not session.get("admin_id"):
            return redirect(
                url_for("admin_login")
            )

        return view(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------
# Student - face login
# ---------------------------------------------------------------------

@app.route("/student/login")
def student_login():
    return render_template(
        "student_face_login.html"
    )


@app.route(
    "/api/student/login",
    methods=["POST"]
)
def api_student_login():

    data = request.get_json(force=True) or {}

    name = (
        data.get("name") or ""
    ).strip()

    faculty = (
        data.get("faculty") or ""
    ).strip()

    image_data = data.get("image")


    # Check name and faculty
    if not name or not faculty:

        return jsonify({
            "ok": False,
            "message": "Name and faculty are required."
        }), 400


    # Find student
    student = db.get_student_by_name_faculty(
        name,
        faculty
    )

    if not student:

        return jsonify({
            "ok": False,
            "message": "Student not found. Check your name/faculty."
        }), 404


    # Check image
    if not image_data:

        return jsonify({
            "ok": False,
            "message": "No photo received."
        }), 400


    # Decode image
    img = decode_base64_image(
        image_data
    )

    if img is None:

        return jsonify({
            "ok": False,
            "message": (
                "Could not read the captured photo. "
                "Please try again."
            )
        }), 400


    try:

        # -------------------------------------------------------------
        # 1. Detect face
        # -------------------------------------------------------------

        face = recognition.get_largest_face(
            img
        )

        if face is None:

            return jsonify({
                "ok": False,
                "message": (
                    "No face detected. "
                    "Please center your face in the camera "
                    "and try again."
                )
            }), 400


        # -------------------------------------------------------------
        # 2. Liveness / anti-spoofing
        # -------------------------------------------------------------

        live, score, engine = liveness.is_live_face(
            img,
            face.bbox
        )

        if not live:

            return jsonify({
                "ok": False,
                "message": (
                    f"Liveness check failed "
                    f"({engine}, score {score:.2f}). "
                    f"A photo cannot be used to log in."
                )
            }), 403


        # -------------------------------------------------------------
        # 3. Load stored face embedding
        # -------------------------------------------------------------

        stored_embedding = json.loads(
            student["descriptor"]
        )


        # -------------------------------------------------------------
        # 4. Compare face embeddings
        # -------------------------------------------------------------

        similarity = recognition.cosine_similarity(
            face.normed_embedding.tolist(),
            stored_embedding
        )


        if similarity < recognition.MATCH_THRESHOLD:

            return jsonify({
                "ok": False,
                "message": (
                    f"Face did not match "
                    f"(similarity {similarity:.2f}). "
                    f"Please try again."
                )
            }), 401


        # -------------------------------------------------------------
        # 5. Login successful
        # -------------------------------------------------------------

        session["student_id"] = student["id"]
        session["student_name"] = student["name"]
        session["student_faculty"] = student["faculty"]


        # Email may be NULL for old students
        if "email" in student.keys():

            session["student_email"] = student["email"]


        return jsonify({
            "ok": True
        })


    except FaceEngineUnavailable as e:

        return jsonify({
            "ok": False,
            "message": str(e)
        }), 500


    except Exception as e:

        print(
            "Student login error:",
            e
        )

        return jsonify({
            "ok": False,
            "message": (
                "Face recognition error. "
                "Please try again."
            )
        }), 500


# ---------------------------------------------------------------------
# Student logout
# ---------------------------------------------------------------------

@app.route("/student/logout")
def student_logout():

    session.clear()

    return redirect(
        url_for("index")
    )


# ---------------------------------------------------------------------
# Student - dashboard
# ---------------------------------------------------------------------

@app.route("/student/dashboard")
@student_required
def student_dashboard():

    return render_template(
        "student_dashboard.html",
        name=session.get("student_name"),
        faculty=session.get("student_faculty"),
        unpaid_count=db.count_unpaid_for_student(
            session.get("student_id")
        ),
    )


# ---------------------------------------------------------------------
# Student - equipment
# ---------------------------------------------------------------------

@app.route("/student/equipment")
@student_required
def student_equipment():

    equipment = db.get_all_equipment()

    return render_template(
        "student_equipment.html",
        equipment=equipment
    )


# ---------------------------------------------------------------------
# Student - borrow
# ---------------------------------------------------------------------

@app.route("/student/borrow")
@student_required
def student_borrow():

    return render_template(
        "scan_qr.html",
        action="borrow"
    )


# ---------------------------------------------------------------------
# Student - return
# ---------------------------------------------------------------------

@app.route("/student/return")
@student_required
def student_return():

    return render_template(
        "scan_qr.html",
        action="return"
    )


# ---------------------------------------------------------------------
# Student - QR scan
# ---------------------------------------------------------------------

@app.route(
    "/api/scan",
    methods=["POST"]
)
@student_required
def api_scan():

    data = request.get_json(force=True) or {}

    code = (
        data.get("code") or ""
    ).strip()

    action = data.get("action")

    equipment = db.get_equipment_by_code(
        code
    )

    if not equipment:

        return jsonify({
            "ok": False,
            "message": (
                "Unknown QR code. "
                "Equipment not found."
            )
        }), 404


    student_id = session.get(
        "student_id"
    )


    # -------------------------------------------------------------
    # Borrow
    # -------------------------------------------------------------

    if action == "borrow":

        if equipment["status"] == "broken":

            return jsonify({
                "ok": False,
                "message": (
                    f"{equipment['name']} "
                    f"is marked as broken and cannot be borrowed."
                )
            }), 400


        if equipment["status"] == "borrowed":

            return jsonify({
                "ok": False,
                "message": (
                    f"{equipment['name']} "
                    f"is already borrowed by someone else."
                )
            }), 400


        db.record_borrow(
            student_id,
            equipment["id"]
        )


        return jsonify({
            "ok": True,
            "message": (
                f"{equipment['name']} "
                f"borrowed successfully."
            ),
            "equipment_name": equipment["name"],
            "code": equipment["code"],
            "action": "borrow"
        })


    # -------------------------------------------------------------
    # Return
    # -------------------------------------------------------------

    elif action == "return":

        active_request = db.get_active_request_for_equipment(
            equipment["id"]
        )


        if active_request is None:

            return jsonify({
                "ok": False,
                "message": (
                    f"{equipment['name']} "
                    f"is not currently borrowed."
                )
            }), 400


        if active_request["student_id"] != student_id:

            return jsonify({
                "ok": False,
                "message": (
                    "This equipment was not borrowed by you."
                )
            }), 403


        db.add_return(
            active_request["id"]
        )


        return jsonify({
            "ok": True,
            "message": (
                f"{equipment['name']} "
                f"returned successfully."
            ),
            "equipment_name": equipment["name"],
            "code": equipment["code"],
            "action": "return"
        })


    return jsonify({
        "ok": False,
        "message": "Invalid action."
    }), 400


# ---------------------------------------------------------------------
# Student - scanned equipment
# ---------------------------------------------------------------------

@app.route("/student/scanned")
@student_required
def scanned_equipment():

    return render_template(
        "scanned_equipment.html",
        equipment_name=request.args.get(
            "equipment_name",
            ""
        ),
        action=request.args.get(
            "action",
            ""
        ),
        status=request.args.get(
            "status",
            ""
        ),
        message=request.args.get(
            "message",
            ""
        ),
    )


# ---------------------------------------------------------------------
# Student - borrowed/history
# ---------------------------------------------------------------------

@app.route("/student/borrowed")
@student_required
def student_borrowed():

    student_id = session.get(
        "student_id"
    )

    return render_template(
        "student_borrowed.html",
        records=db.get_records_by_student(
            student_id
        ),
        qr_file=db.get_setting(
            "payment_qr"
        )
    )


# ---------------------------------------------------------------------
# Student - wishlist
# ---------------------------------------------------------------------

@app.route(
    "/student/request-equipment",
    methods=["GET", "POST"]
)
@student_required
def student_request_equipment():

    if request.method == "POST":

        item_name = request.form.get(
            "item_name",
            ""
        ).strip()

        notes = request.form.get(
            "notes",
            ""
        ).strip()


        if item_name:

            db.add_wishlist_request(
                session.get("student_id"),
                item_name,
                notes
            )


        return redirect(
            url_for(
                "student_request_equipment",
                submitted=1
            )
        )


    return render_template(
        "request_equipment.html",
        submitted=request.args.get(
            "submitted"
        )
    )


# ---------------------------------------------------------------------
# Student - Payment submission
# ---------------------------------------------------------------------

@app.route(
    "/student/pay/<int:payment_id>/paid",
    methods=["POST"]
)
@student_required
def student_pay_submit(payment_id):

    # Transaction/reference ID
    reference = request.form.get(
        "reference",
        ""
    ).strip()[:100]


    # -------------------------------------------------------------
    # Get payment screenshot
    # -------------------------------------------------------------

    screenshot = request.files.get(
        "payment_screenshot"
    )


    if not screenshot or not screenshot.filename:

        flash(
            "Please upload your payment screenshot.",
            "error"
        )

        return redirect(
            url_for("student_borrowed")
        )


    file_bytes = screenshot.read()


    if not file_bytes:

        flash(
            "The payment screenshot is empty.",
            "error"
        )

        return redirect(
            url_for("student_borrowed")
        )


    # -------------------------------------------------------------
    # Check actual image type
    # -------------------------------------------------------------

    real_ext = _image_extension(
        file_bytes
    )


    if real_ext not in ALLOWED_QR_EXTENSIONS:

        flash(
            "Only PNG, JPG, WEBP or GIF images are allowed.",
            "error"
        )

        return redirect(
            url_for("student_borrowed")
        )


    # -------------------------------------------------------------
    # Create safe unique filename
    # -------------------------------------------------------------

    filename = (
        f"payment_proof_"
        f"{uuid.uuid4().hex}."
        f"{real_ext}"
    )


    proof_path = os.path.join(
        PAYMENT_PROOF_DIR,
        filename
    )


    # -------------------------------------------------------------
    # Save screenshot
    # -------------------------------------------------------------

    with open(
        proof_path,
        "wb"
    ) as out:

        out.write(
            file_bytes
        )


    # -------------------------------------------------------------
    # Save payment submission in database
    # -------------------------------------------------------------

    ok = db.student_mark_payment_submitted(
        payment_id,
        session.get("student_id"),
        reference,
        filename
    )


    if ok:

        flash(
            "Payment submitted successfully. "
            "The admin will check your payment screenshot.",
            "success"
        )


    else:

        # Remove screenshot if database update failed.
        if os.path.isfile(proof_path):

            os.remove(
                proof_path
            )


        flash(
            "This payment could not be updated.",
            "error"
        )


    return redirect(
        url_for("student_borrowed")
    )


# =====================================================================
# ADMIN
# =====================================================================

# ---------------------------------------------------------------------
# Admin login
# ---------------------------------------------------------------------

@app.route(
    "/admin/login",
    methods=["GET", "POST"]
)
def admin_login():

    error = None


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        admin = db.get_admin_by_username(
            username
        )


        if admin and check_password_hash(
            admin["password_hash"],
            password
        ):

            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]

            return redirect(
                url_for("admin_dashboard")
            )


        error = "Invalid username or password."


    return render_template(
        "admin_login.html",
        error=error
    )


# ---------------------------------------------------------------------
# Admin logout
# ---------------------------------------------------------------------

@app.route("/admin/logout")
def admin_logout():

    session.pop(
        "admin_id",
        None
    )

    session.pop(
        "admin_username",
        None
    )

    return redirect(
        url_for("index")
    )


# ---------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------

@app.route("/admin")
@admin_required
def admin_dashboard():

    equipment = db.get_all_equipment()

    students = db.get_all_students()

    wishlist = db.get_all_wishlist_requests()


    available = sum(
        1
        for e in equipment
        if e["status"] == "available"
    )


    borrowed = sum(
        1
        for e in equipment
        if e["status"] == "borrowed"
    )


    return render_template(
        "admin_dashboard.html",
        total_equipment=len(equipment),
        available=available,
        borrowed=borrowed,
        total_students=len(students),
        wishlist_count=len(wishlist),
        broken_count=sum(
            1
            for e in equipment
            if e["status"] == "broken"
        ),
        pending_payments=db.count_pending_payments(),
    )


# ---------------------------------------------------------------------
# Admin - equipment
# ---------------------------------------------------------------------

@app.route("/admin/equipment")
@admin_required
def admin_equipment():

    equipment = db.get_all_equipment()

    return render_template(
        "equipment.html",
        equipment=equipment
    )


@app.route(
    "/admin/equipment/add",
    methods=["GET", "POST"]
)
@admin_required
def admin_equipment_add():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        category = request.form.get(
            "category",
            ""
        ).strip()


        if name and category:

            db.add_equipment(
                name,
                category
            )


        return redirect(
            url_for("admin_equipment")
        )


    return render_template(
        "add_equipment.html"
    )


@app.route(
    "/admin/equipment/delete/<int:equipment_id>",
    methods=["POST"]
)
@admin_required
def admin_equipment_delete(
    equipment_id
):

    try:

        db.delete_equipment(
            equipment_id
        )

    except sqlite3.IntegrityError:

        flash(
            "This equipment has borrow history "
            "and cannot be deleted.",
            "error"
        )

    return redirect(
        url_for("admin_equipment")
    )


# ---------------------------------------------------------------------
# Admin - requests
# ---------------------------------------------------------------------

@app.route("/admin/requests")
@admin_required
def admin_requests():

    wishlist = db.get_all_wishlist_requests()

    return render_template(
        "equipment_requests.html",
        wishlist=wishlist
    )


@app.route(
    "/admin/requests/<int:wishlist_id>/delete",
    methods=["POST"]
)
@admin_required
def admin_wishlist_delete(
    wishlist_id
):

    db.delete_wishlist_request(
        wishlist_id
    )

    return redirect(
        url_for("admin_requests")
    )


# ---------------------------------------------------------------------
# Admin - reports
# ---------------------------------------------------------------------

@app.route("/admin/reports")
@admin_required
def admin_reports():

    report_rows = db.get_all_requests_report()

    summary = db.get_report_summary()

    return render_template(
        "reports.html",
        rows=report_rows,
        summary=summary
    )


# =====================================================================
# ADMIN - STUDENT MANAGEMENT
# =====================================================================

# ---------------------------------------------------------------------
# Student list
# ---------------------------------------------------------------------

@app.route("/admin/students")
@admin_required
def admin_students():

    students = db.get_all_students()

    return render_template(
        "students.html",
        students=students
    )


# ---------------------------------------------------------------------
# Student face registration page
# ---------------------------------------------------------------------

@app.route("/admin/students/register")
@admin_required
def admin_students_register():

    name = request.args.get(
        "name",
        ""
    ).strip()

    faculty = request.args.get(
        "faculty",
        ""
    ).strip()

    email = request.args.get(
        "email",
        ""
    ).strip().lower()


    return render_template(
        "student_face.html",
        name=name,
        faculty=faculty,
        email=email
    )


# ---------------------------------------------------------------------
# Admin-only student registration
# ---------------------------------------------------------------------

@app.route(
    "/api/student/register",
    methods=["POST"]
)
@admin_required
def api_student_register():

    data = request.get_json(
        force=True
    ) or {}


    name = (
        data.get("name") or ""
    ).strip()

    faculty = (
        data.get("faculty") or ""
    ).strip()

    email = (
        data.get("email") or ""
    ).strip().lower()

    image_data = data.get(
        "image"
    )


    # -------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------

    if not name:

        return jsonify({
            "ok": False,
            "message": "Student name is required."
        }), 400


    if not faculty:

        return jsonify({
            "ok": False,
            "message": "Faculty is required."
        }), 400


    if not email:

        return jsonify({
            "ok": False,
            "message": "Email is required."
        }), 400


    if not image_data:

        return jsonify({
            "ok": False,
            "message": "Student face is required."
        }), 400


    # -------------------------------------------------------------
    # Email validation
    # -------------------------------------------------------------

    email_pattern = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"


    if not re.fullmatch(
        email_pattern,
        email
    ):

        return jsonify({
            "ok": False,
            "message": (
                "Please enter a valid email address."
            )
        }), 400


    # -------------------------------------------------------------
    # Duplicate name + faculty
    # -------------------------------------------------------------

    if db.get_student_by_name_faculty(
        name,
        faculty
    ):

        return jsonify({
            "ok": False,
            "message": (
                "A student with this name "
                "and faculty already exists."
            )
        }), 400


    # -------------------------------------------------------------
    # Duplicate email
    # -------------------------------------------------------------

    if hasattr(
        db,
        "get_student_by_email"
    ):

        if db.get_student_by_email(
            email
        ):

            return jsonify({
                "ok": False,
                "message": (
                    "This email address is already "
                    "registered to another student."
                )
            }), 400


    # -------------------------------------------------------------
    # Decode image
    # -------------------------------------------------------------

    img = decode_base64_image(
        image_data
    )


    if img is None:

        return jsonify({
            "ok": False,
            "message": (
                "Could not read the captured photo. "
                "Please try again."
            )
        }), 400


    try:

        # ---------------------------------------------------------
        # 1. Detect face
        # ---------------------------------------------------------

        face = recognition.get_largest_face(
            img
        )


        if face is None:

            return jsonify({
                "ok": False,
                "message": (
                    "No face detected. "
                    "Please center the student's "
                    "face in the camera and try again."
                )
            }), 400


        # ---------------------------------------------------------
        # 2. Liveness check
        # ---------------------------------------------------------

        live, score, engine = (
            liveness.is_live_face(
                img,
                face.bbox
            )
        )


        if not live:

            return jsonify({
                "ok": False,
                "message": (
                    f"Liveness check failed "
                    f"({engine}, score {score:.2f}). "
                    f"Please use a real live face, "
                    f"not a photo."
                )
            }), 400


        # ---------------------------------------------------------
        # 3. Save face embedding + student information
        # ---------------------------------------------------------

        db.add_student(
            name,
            faculty,
            email,
            face.normed_embedding.tolist()
        )


        return jsonify({
            "ok": True,
            "message": (
                "Student registered successfully."
            )
        })


    except FaceEngineUnavailable as e:

        return jsonify({
            "ok": False,
            "message": str(e)
        }), 500


    except sqlite3.IntegrityError:

        return jsonify({
            "ok": False,
            "message": (
                "This student or email is already registered."
            )
        }), 400


    except Exception as e:

        print(
            "Student registration error:",
            e
        )

        return jsonify({
            "ok": False,
            "message": (
                "An error occurred while "
                "registering the student."
            )
        }), 500


# ---------------------------------------------------------------------
# Delete student
# ---------------------------------------------------------------------

@app.route(
    "/admin/students/delete/<int:student_id>",
    methods=["POST"]
)
@admin_required
def admin_students_delete(
    student_id
):

    try:

        db.delete_student(
            student_id
        )

    except sqlite3.IntegrityError:

        flash(
            "This student has borrow or payment "
            "history and cannot be deleted.",
            "error"
        )

    return redirect(
        url_for("admin_students")
    )


# =====================================================================
# ADMIN - EQUIPMENT CONDITION
# =====================================================================

@app.route(
    "/admin/equipment/<int:equipment_id>/status",
    methods=["POST"]
)
@admin_required
def admin_equipment_status(
    equipment_id
):

    new_status = request.form.get(
        "status",
        ""
    )

    equipment = db.get_equipment_by_id(
        equipment_id
    )


    if (
        not equipment
        or new_status not in (
            "available",
            "broken"
        )
    ):

        return redirect(
            url_for("admin_equipment")
        )


    if equipment["status"] == "borrowed":

        flash(
            "This equipment is currently borrowed. "
            "Wait until it is returned first.",
            "error"
        )

    else:

        db.update_equipment_status(
            equipment_id,
            new_status
        )


    return redirect(
        url_for("admin_equipment")
    )


# =====================================================================
# ADMIN - PAYMENTS + PAYMENT QR UPLOAD
# =====================================================================

def _image_extension(file_bytes):
    """
    Detects the real image type from the file header,
    not only from the filename.
    """

    if file_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        return "png"


    if file_bytes.startswith(
        b"\xff\xd8\xff"
    ):
        return "jpg"


    if file_bytes[:6] in (
        b"GIF87a",
        b"GIF89a"
    ):
        return "gif"


    if (
        file_bytes[:4] == b"RIFF"
        and file_bytes[8:12] == b"WEBP"
    ):
        return "webp"


    return None


def _delete_payment_qr_file():

    old_name = db.get_setting(
        "payment_qr"
    )


    if old_name:

        old_path = os.path.join(
            UPLOAD_DIR,
            os.path.basename(old_name)
        )


        if os.path.isfile(
            old_path
        ):

            os.remove(
                old_path
            )


@app.route("/admin/payments")
@admin_required
def admin_payments():

    return render_template(
        "payments.html",
        payments=db.get_all_payments(),
        qr_file=db.get_setting(
            "payment_qr"
        ),
        charge_options=db.get_requests_without_charge()
    )


@app.route(
    "/admin/payments/add",
    methods=["POST"]
)
@admin_required
def admin_payment_add():

    try:

        request_id = int(
            request.form.get(
                "request_id",
                ""
            )
        )

        amount = round(
            float(
                request.form.get(
                    "amount",
                    ""
                )
            ),
            2
        )


    except ValueError:

        flash(
            "Please choose a borrow record and enter a valid amount.",
            "error"
        )

        return redirect(
            url_for("admin_payments")
        )


    if amount <= 0 or amount > 10000000:

        flash(
            "Please enter an amount greater than 0.",
            "error"
        )

        return redirect(
            url_for("admin_payments")
        )


    result = db.add_charge(
        request_id,
        amount,
        request.form.get(
            "note",
            ""
        ).strip()[:500]
    )


    if result == "ok":

        flash(
            "Charge added. "
            "The student will see it in My Borrowed Items.",
            "success"
        )


    elif result == "exists":

        flash(
            "That borrow already has a charge.",
            "error"
        )


    else:

        flash(
            "Borrow record not found.",
            "error"
        )


    return redirect(
        url_for("admin_payments")
    )


@app.route(
    "/admin/payments/qr",
    methods=["POST"]
)
@admin_required
def admin_payment_qr_upload():

    file = request.files.get(
        "qr_image"
    )


    if not file or not file.filename:

        flash(
            "Please choose an image file.",
            "error"
        )

        return redirect(
            url_for("admin_payments")
        )


    file_bytes = file.read()


    declared_ext = (
        file.filename
        .rsplit(".", 1)[-1]
        .lower()
    )


    real_ext = _image_extension(
        file_bytes
    )


    if (
        declared_ext not in ALLOWED_QR_EXTENSIONS
        or real_ext is None
    ):

        flash(
            "Only PNG, JPG, WEBP or GIF images are allowed.",
            "error"
        )

        return redirect(
            url_for("admin_payments")
        )


    _delete_payment_qr_file()


    # Timestamp stops browsers showing a cached old QR.
    filename = (
        f"payment_qr_"
        f"{int(time.time())}."
        f"{real_ext}"
    )


    with open(
        os.path.join(
            UPLOAD_DIR,
            filename
        ),
        "wb"
    ) as out:

        out.write(
            file_bytes
        )


    db.set_setting(
        "payment_qr",
        filename
    )


    flash(
        "Payment QR uploaded. Students can now see it.",
        "success"
    )


    return redirect(
        url_for("admin_payments")
    )


@app.route(
    "/admin/payments/qr/delete",
    methods=["POST"]
)
@admin_required
def admin_payment_qr_delete():

    _delete_payment_qr_file()

    db.set_setting(
        "payment_qr",
        None
    )

    flash(
        "Payment QR removed.",
        "success"
    )

    return redirect(
        url_for("admin_payments")
    )


@app.route(
    "/admin/payments/<int:payment_id>/confirm",
    methods=["POST"]
)
@admin_required
def admin_payment_confirm(
    payment_id
):

    if db.admin_confirm_payment(
        payment_id
    ):

        flash(
            "Payment marked as paid.",
            "success"
        )


    return redirect(
        url_for("admin_payments")
    )


@app.route(
    "/admin/payments/<int:payment_id>/cancel",
    methods=["POST"]
)
@admin_required
def admin_payment_cancel(
    payment_id
):

    if db.admin_cancel_payment(
        payment_id
    ):

        flash(
            "Charge removed.",
            "success"
        )

    else:

        flash(
            "A paid charge cannot be removed.",
            "error"
        )


    return redirect(
        url_for("admin_payments")
    )


# ---------------------------------------------------------------------
# Nav badges + friendly error for oversized uploads
# ---------------------------------------------------------------------

@app.context_processor
def inject_admin_nav_counts():

    if session.get("admin_id"):

        return {
            "nav_pending_payments":
                db.count_pending_payments(),
        }


    return {}


@app.errorhandler(413)
def file_too_large(error):

    flash(
        "That file is too large (maximum 4 MB).",
        "error"
    )


    if session.get("admin_id"):

        return redirect(
            url_for("admin_payments")
        )


    return redirect(
        url_for("index")
    )


# ---------------------------------------------------------------------
# Run application
# ---------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )