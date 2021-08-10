SELECT publisher_name
FROM publisher
WHERE publisher_code NOT IN
	(SELECT publisher_code
     FROM copy JOIN book USING (book_code));

-- list of publishers whose books are in the DB     
SELECT distinct publisher_name
FROM publisher JOIN book USING (publisher_code);

-- list of publishers whose books are available for purchase in a branch
SELECT distinct publisher_name
FROM copy JOIN book USING (book_code)
		  JOIN publisher USING (publisher_code);

-- list of publishers in the DB whose books are unavailable for puchase in a branch
SELECT distinct publisher_name
FROM publisher JOIN book USING (publisher_code)
WHERE publisher_code NOT IN
	(SELECT publisher_code
    FROM copy JOIN book USING (book_code));