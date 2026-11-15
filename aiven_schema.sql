-- ============================================================
-- Smart Attendance System - Non-destructive Aiven MySQL import
-- MySQL 8.x / Aiven MySQL
--
-- Run this only after selecting the intended Aiven database in your
-- SQL client or connection configuration. This script does not create
-- or select a database, and it does not drop any existing objects.
-- ============================================================

-- ------------------------------------------------------------
-- Table: students
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    roll_number   VARCHAR(20)  NOT NULL,
    name          VARCHAR(100) NOT NULL,
    password      VARCHAR(255) NOT NULL,
    face_data     LONGTEXT     NULL,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_students_roll_number UNIQUE (roll_number)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Table: teachers
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teachers (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL,
    name          VARCHAR(100) NOT NULL,
    password      VARCHAR(255) NOT NULL,
    role          VARCHAR(100) NOT NULL DEFAULT 'Faculty',
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_teachers_username UNIQUE (username)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Table: attendance
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    student_id    INT UNSIGNED NOT NULL,
    date          DATE         NOT NULL,
    time          TIME         NOT NULL,
    latitude      DECIMAL(10, 8) NOT NULL,
    longitude     DECIMAL(11, 8) NOT NULL,
    status        ENUM('Present', 'Absent', 'Late') NOT NULL DEFAULT 'Present',
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_attendance_student
        FOREIGN KEY (student_id) REFERENCES students(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT uq_attendance_student_date UNIQUE (student_id, date),
    INDEX idx_attendance_student_id (student_id),
    INDEX idx_attendance_date (date),
    INDEX idx_attendance_status (status)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Non-destructive seed data
-- ------------------------------------------------------------
-- Resolve each attendance row through roll_number so it references the
-- correct cloud student id even when Aiven generated different ids.

-- Demo account and attendance seeds are intentionally omitted. Provision accounts securely.
