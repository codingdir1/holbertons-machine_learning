DELIMITER //

DROP PROCEDURE IF EXISTS AddBonus;
-- creates a stored procedure AddBonus that adds
-- a new correction for a student

CREATE PROCEDURE AddBonus (
	IN user_id INT, 
	IN project_name VARCHAR(255), 
	IN score INT)
BEGIN

	DECLARE project_id INT;
	INSERT INTO projects (name)
	SELECT project_name
	WHERE NOT EXISTS(
		SELECT 1
		FROM projects
		WHERE name = project_name
	);

	SELECT id INTO project_id
	FROM projects
	WHERE name = project_name
	LIMIT 1;

	INSERT corrections (user_id, project_id, score)
	SELECT user_id, project_id, score
	WHERE NOT EXISTS(
		SELECT 1
		FROM corrections
		WHERE user_id = corrections.user_id AND project_id = corrections.project_id
	);

	UPDATE corrections
	SET corrections.score = score
	WHERE corrections.user_id = user_id AND corrections.project_id = project_id;
END //

DELIMITER ;
