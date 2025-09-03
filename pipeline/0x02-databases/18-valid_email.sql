DELIMITER //

DROP TRIGGER IF EXISTS reset_valid_email;

CREATE TRIGGER reset_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
	IF NOT (NEW.email = OLD.email) THEN
		SET NEW.valid_email = 0;
	END IF;
END//

DELIMITER ;
