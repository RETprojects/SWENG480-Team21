CREATE TABLE pattern_catalog
(
  id INT(5) NOT NULL AUTO_INCREMENT,
  name VARCHAR(150) NOT NULL,
  description VARCHAR(300),
  url VARCHAR(2000) NOT NULL,
  PRIMARY KEY (id)
);

CREATE TABLE pattern_category
(
  id INT(10) NOT NULL AUTO_INCREMENT,
  catalog_id INT(5) NOT NULL,	
  name VARCHAR(150) NOT NULL,
  description VARCHAR(300) NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (catalog_id) REFERENCES pattern_catalog(id)
);

CREATE TABLE pattern
(
  id INT(20) NOT NULL AUTO_INCREMENT,
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
  PRIMARY KEY (id),
  FOREIGN KEY (category_id) REFERENCES pattern_category(id)
);

CREATE TABLE pattern_ML
(
  id INT(20) NOT NULL AUTO_INCREMENT,
  category_id INT(10) NOT NULL,
  name VARCHAR(150) NOT NULL,
  intent VARCHAR(1000) NOT NULL,
  problem VARCHAR(1000) NOT NULL,
  discussion VARCHAR(1000) NOT NULL,
  structure VARCHAR(1000) NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (category_id) REFERENCES pattern_category(id)
);

CREATE TABLE problem
(
  id INT(50) NOT NULL AUTO_INCREMENT,
  category_id INT(10) NOT NULL,
  description VARCHAR(150) NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (category_id) REFERENCES pattern_category(id)
);

CREATE TABLE problem_pattern_match
(
  id INT(200) NOT NULL AUTO_INCREMENT,
  problem_id INT(50) NOT NULL,
  pattern_id INT(20) NOT NULL,
  similarity_score DOUBLE(23, 20),
  PRIMARY KEY (id),
  FOREIGN KEY (problem_id) REFERENCES problem(id),
  FOREIGN KEY (pattern_id) REFERENCES pattern(id)
);
