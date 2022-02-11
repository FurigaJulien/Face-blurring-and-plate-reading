
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR NOT NULL,
    password VARCHAR NOT NULL,
    first_name VARCHAR NOT NULL,
    family_name VARCHAR NOT NULL
);

CREATE TABLE plates (
    id SERIAL PRIMARY KEY,
    user_id SERIAL,
    plate_number VARCHAR NOT NULL,
    CONSTRAINT fk_user
        FOREIGN KEY (user_id)
            REFERENCES users(id)
);
username