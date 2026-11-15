-- Smart Attendance face-template storage migration.
-- Review and run manually against the selected Aiven database only after backup.
-- This file has not been executed by the implementation task.

CREATE TABLE IF NOT EXISTS student_face_embeddings (
    id             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    student_id     INT UNSIGNED NOT NULL,
    embedding      VARBINARY(576) NOT NULL,
    model_name     VARCHAR(64) NOT NULL,
    model_version  VARCHAR(128) NOT NULL,
    preprocessing_version VARCHAR(128) NOT NULL
        DEFAULT 'retinaface5pt112-rgb01-imagenetnorm-tta-v1',
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_face_embeddings_student_model (student_id, model_name, model_version),
    CONSTRAINT fk_face_embeddings_student
        FOREIGN KEY (student_id) REFERENCES students(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;
