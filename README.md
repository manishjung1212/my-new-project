# SEMS - Sport Equipment Management System

A Flask web app to manage borrowing and returning of sport equipment
(football, badminton, basketball, etc.). Students log in with **face
recognition + liveness check**, equipment is tracked with **QR codes**,
and the admin manages students, broken equipment and payments.

## What's new in this version

- **No student self-registration.** Only the admin can register a student
  (name, faculty, email and a live face capture).
- **Broken equipment is handled by the admin.** If something breaks, the
  student simply **returns it by QR code** as usual. The admin then marks
  that equipment **Broken** (Admin -> Equipment). Broken equipment cannot be
  borrowed until the admin marks it **Repaired**.
- **My Borrowed Items** shows, for every borrow: **borrowed time, return
  time, status and pay status**.
- **Pay option inside My Borrowed Items** (no separate page). The admin
  uploads a payment QR and adds a charge to a borrow; the student presses
  **Pay**, scans the admin's QR, then presses **I have paid**. The admin
  confirms.
- All buttons use the same colour.

## How it works

### Student

1. Click **Student Login**, enter name + faculty, then verify your face
   (liveness check + face match). If you are not registered, ask the admin.
2. Student Dashboard:
   - **View Equipment** - all equipment and its status
     (`available` / `borrowed` / `broken`).
   - **Borrow / Return Equipment** - scan the item's QR code.
   - **My Borrowed Items** - borrowed time, return time, pay status, and the
     **Pay** button for any charge the admin added.
   - **Request New Equipment** - ask the admin to add equipment.

### Admin

Default login (change it, see Notes): `admin` / `admin123`

- **Equipment** - add / delete equipment, download QR codes, and
  **Mark Broken / Mark Repaired**.
- **Students** - register new students (the only way to create students).
- **Requests** - equipment students have asked for.
- **Payments** - upload / replace / remove the **payment QR**, **add a
  charge** to a borrow record (amount + note), and **Confirm Paid** or
  remove charges.
- **Reports** - borrow history plus broken-equipment and payment totals.

## Project structure

```
SEMS/
├── app.py
├── requirements.txt
├── face_engine/                # your own package (recognition + liveness)
├── database/
│   ├── __init__.py
│   └── db.py                   # SQLite helpers (auto-upgrades old sems.db)
├── static/
│   ├── css/style.css
│   ├── img/
│   └── uploads/                # admin-uploaded payment QR image
└── templates/
    ├── index.html
    ├── student_face_login.html
    ├── student_dashboard.html
    ├── student_equipment.html
    ├── scan_qr.html / scanned_equipment.html
    ├── student_borrowed.html   # borrowed/returned times, pay status, Pay popup
    ├── request_equipment.html
    ├── admin_login.html / admin_dashboard.html
    ├── _admin_nav.html / _flash.html   # shared pieces
    ├── students.html / student_face.html   # admin: register students
    ├── equipment.html / add_equipment.html
    ├── equipment_requests.html
    ├── payments.html           # admin: payment QR upload, add charge, confirm
    └── reports.html
```

## Data model

- **students** - name, faculty, email, face embedding
- **equipment** - items with a unique QR code; status `available` /
  `borrowed` / `broken`
- **requests** - one row per borrow (`approved` -> `returned`)
- **returns** - completed returns, linked 1:1 to a request
- **payments** - a charge on a borrow record: `due` -> `submitted` (student
  says paid) -> `paid` (admin confirmed); optional note and reference
- **settings** - key/value (stores the payment QR filename)
- **wishlist**, **admins**

## Setup

```bash
cd SEMS
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**. `sems.db` is created automatically; an
existing `sems.db` is upgraded in place (nothing is deleted).

## Notes

- **Currency:** change `CURRENCY` at the top of `app.py` (default `Rs.`).
- **Payments are manual:** the app shows your QR and records who says they
  paid; it cannot verify the transfer itself, so check your bank/wallet
  before pressing **Confirm Paid**.
- **Payment QR upload:** only real PNG / JPG / WEBP / GIF files up to 4 MB
  are accepted (the file header is checked, not just the name).
- **History is protected:** equipment or students that already have borrow
  or payment history cannot be deleted (a message is shown). Leave retired
  items marked **Broken** instead.
- **Upgrading:** an existing `sems.db` is upgraded automatically. Charges
  created with the previous version are kept and linked to their borrow.
- Change the default admin password: delete `sems.db` and edit the seeded
  credentials in `database/db.py` (`init_db`) before first run, or update
  the hash directly in the `admins` table.
- Change `app.secret_key` in `app.py` before real use, and set
  `debug=False` in production.
