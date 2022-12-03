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
  id INT(10) unsigned NOT NULL AUTO_INCREMENT,
  catalog_id INT(5) unsigned NOT NULL,	
  name VARCHAR(150) NOT NULL,
  description VARCHAR(300) NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (catalog_id) REFERENCES pattern_catalog(id)
);

CREATE TABLE pattern
(
  id INT(20) unsigned NOT NULL AUTO_INCREMENT,
  category_id INT(10) unsigned NOT NULL,
  name VARCHAR(150) NOT NULL,
  intent VARCHAR(150) NOT NULL,
  motivation VARCHAR(150) NOT NULL,
  applicability VARCHAR(150) NOT NULL,
  structure IMAGE NOT NULL,
  participant_collaborations VARCHAR(150) NOT NULL,
  consequences VARCHAR(150) NOT NULL,
  implementation VARCHAR(150) NOT NULL,
  related_patterns VARCHAR(150) NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (category_id)
);

CREATE TABLE problem
(
  id			INT(50) unsigned NOT NULL AUTO_INCREMENT, # Unique ID for the record
  category_id	INT(10) unsigned NOT NULL,				  # ID for the pattern category to which the problem belongs
  description	VARCHAR(150) NOT NULL,                # Description of the problem
  PRIMARY KEY	(id),                                 # Make the id the primary key
  FOREIGN KEY	(category_id)							  # Make the category id a foreign key
);

CREATE TABLE problem_pattern_match
(
  id					INT(200) unsigned NOT NULL AUTO_INCREMENT, # Unique ID for the record
  problem_id			INT(50) unsigned NOT NULL,				  # ID for the problem that has been matched with a pattern
  pattern_id			INT(20) unsigned NOT NULL,				  # ID for the pattern that has been matched with a problem
  similarity_score	DOUBLE(23),							  # Similarity score of the pattern for the problem
  PRIMARY KEY	(id),                                  		  # Make the id the primary key
  FOREIGN KEY	(problem_id),									  # Make the problem id a foreign key
  FOREIGN KEY	(pattern_id)									  # Make the pattern id another foreign key
);
