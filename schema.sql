-- ============================================================
-- Smart Attendance System - Database Schema
-- MySQL 8.x
-- ============================================================

CREATE DATABASE IF NOT EXISTS smart_attendance
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE smart_attendance;

-- ------------------------------------------------------------
-- Table: students
-- ------------------------------------------------------------
DROP TABLE IF EXISTS attendance;
DROP TABLE IF EXISTS students;

CREATE TABLE students (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    roll_number   VARCHAR(20)  NOT NULL,
    name          VARCHAR(100) NOT NULL,
    password      VARCHAR(255) NOT NULL,   -- store a hashed password, not plain text
    face_data     LONGTEXT     NULL,       -- placeholder for encoded face embedding / descriptor
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_students_roll_number UNIQUE (roll_number)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Table: teachers
-- ------------------------------------------------------------
-- Kept separate from students because the two roles authenticate differently.
CREATE TABLE IF NOT EXISTS teachers (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL,
    name          VARCHAR(100) NOT NULL,
    password      VARCHAR(255) NOT NULL, -- PBKDF2-SHA256 encoded password
    role          VARCHAR(100) NOT NULL DEFAULT 'Faculty',
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_teachers_username UNIQUE (username)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Table: attendance
-- ------------------------------------------------------------
CREATE TABLE attendance (
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

    -- one attendance record per student per day
    CONSTRAINT uq_attendance_student_date UNIQUE (student_id, date)
) ENGINE=InnoDB;

-- Indexes to speed up common lookups (roll_number already indexed via UNIQUE above)
CREATE INDEX idx_attendance_student_id ON attendance (student_id);
CREATE INDEX idx_attendance_date       ON attendance (date);
CREATE INDEX idx_attendance_status     ON attendance (status);

-- ------------------------------------------------------------
-- Sample data
-- ------------------------------------------------------------


-- Demo account and attendance seeds are intentionally omitted. Provision accounts securely.
