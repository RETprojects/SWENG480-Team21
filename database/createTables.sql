CREATE TABLE pattern_catalog
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(150) NOT NULL,
  description VARCHAR(300),
  url VARCHAR(2000) NOT NULL
);

CREATE TABLE pattern_category
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  catalog_id INT(5) NOT NULL,	
  name VARCHAR(150) NOT NULL,
  description VARCHAR(300) NOT NULL,
  FOREIGN KEY (catalog_id) REFERENCES pattern_catalog(id)
);

CREATE TABLE pattern
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id INT(10) NOT NULL,
  name VARCHAR(150) NOT NULL,
  intent VARCHAR(150) NOT NULL,
  motivation VARCHAR(150) NOT NULL,
  applicability VARCHAR(150) NOT NULL,
  structure MEDIUMBLOB NOT NULL,
  participant_collaborations VARCHAR(150) NOT NULL,
  consequences VARCHAR(150) NOT NULL,
  implementation VARCHAR(150) NOT NULL,
  related_patterns VARCHAR(150) NOT NULL,
  FOREIGN KEY (category_id) REFERENCES pattern_category(id)
);

CREATE TABLE pattern_ML
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id INT(10) NOT NULL,
  name VARCHAR(150) NOT NULL,
  intent VARCHAR(1000) NOT NULL,
  problem VARCHAR(1000) NOT NULL,
  discussion VARCHAR(1000) NOT NULL,
  structure VARCHAR(1000) NOT NULL,
  FOREIGN KEY (category_id) REFERENCES pattern_category(id)
);

CREATE TABLE problem
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id INT(10) NOT NULL,
  description VARCHAR(150) NOT NULL,
  FOREIGN KEY (category_id) REFERENCES pattern_category(id)
);

CREATE TABLE problem_pattern_match
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  problem_id INT(50) NOT NULL,
  pattern_id INT(20) NOT NULL,
  similarity_score DOUBLE(23, 20),
  FOREIGN KEY (problem_id) REFERENCES problem(id),
  FOREIGN KEY (pattern_id) REFERENCES pattern(id)
);
