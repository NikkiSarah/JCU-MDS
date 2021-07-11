-- Q1
SELECT publisher_name
FROM publisher
WHERE publisher_city <> 'New York';

-- Q2
SELECT book_title AS 'title'
FROM book
WHERE publisher_code =
	(SELECT publisher_code
     FROM publisher
     WHERE publisher_name = 'Penguin USA');

-- Q2 alt using a join
SELECT book_title AS 'title'
FROM publisher JOIN book USING (publisher_code)
WHERE publisher_name = 'Penguin USA';

-- Q3
SELECT book_title AS 'title'
FROM book
WHERE book_type = 'SFI'
	  AND book_paperback = 'TRUE';

-- Q4
SELECT book_title AS 'title'
FROM book
WHERE book_type IN ('CMP', 'HIS', 'SCI');

-- Q5
SELECT count(*) AS 'book count'
FROM author JOIN wrote USING (author_num)
WHERE (author_first = 'Dick' AND author_last = 'Francis')
	  OR (author_first = 'Vernor' AND author_last = 'Vintage');

-- Q5 alt using a subquery
SELECT count(*) AS 'book count'
FROM wrote
WHERE author_num IN
	(SELECT author_num
     FROM author
     WHERE (author_first = 'Dick' AND author_last = 'Francis')
		   OR (author_first = 'Vernor' AND author_last = 'Vintage'));      

-- Q6
SELECT book_title AS 'title'
FROM book JOIN wrote USING (book_code)
		  JOIN author USING (author_num)
WHERE book_type = 'FIC'
	  AND author_first = 'John'
      AND author_last = 'Steinbeck';

-- Q6 alt using a join-subquery combo
SELECT book_title AS 'title'
FROM book JOIN wrote USING (book_code)
WHERE book_type = 'FIC'
	  AND author_num =
		  (SELECT author_num
		  FROM author
          WHERE author_first = 'John'
				AND author_last = 'Steinbeck');

-- Q6 alt using subqueries
SELECT book_title AS 'title'
FROM book
WHERE book_type = 'FIC'
	  AND book_code IN
		  (SELECT book_code
		   FROM wrote
           WHERE author_num =
				 (SELECT author_num
                  FROM author
                  WHERE author_first = 'John'
						AND author_last = 'Steinbeck'));

-- Q7
SELECT count(*) AS 'book count'
FROM copy JOIN branch USING (branch_num)
WHERE branch_name = 'JM Downtown'
	  AND copy_price > 10
      AND copy_price < 20;

-- Q7 alt using a subquery
SELECT count(*) AS 'book count'
FROM copy
WHERE branch_num =
	(SELECT branch_num
     FROM branch
     WHERE branch_name = 'JM Downtown')
	AND copy_price > 10
    AND copy_price < 20;

-- Q8
SELECT branch_name,
	   copy_num,
	   copy_quality AS 'quality',
       copy_price AS 'price'
FROM copy JOIN branch USING (branch_num)
WHERE book_code =
	(SELECT book_code
     FROM book
     WHERE book_title = 'The Stranger');

-- Q8 alt using joins
SELECT branch_name,
	   copy_num,
	   copy_quality AS 'quality',
       copy_price AS 'price'
FROM branch JOIN copy USING (branch_num)
			JOIN book USING (book_code)
WHERE book_title = 'The Stranger';

-- Q9
SELECT book_title AS 'title',
	   count(*) AS 'count',	
	   concat('$', round(avg(copy_price), 2)) AS 'average price'
FROM book JOIN copy USING (book_code)
GROUP BY book_title
	HAVING count(*) > 4
ORDER BY book_title;

-- Q10
SELECT book_title AS 'title',
	   author_first,
	   author_last
FROM branch JOIN copy USING (branch_num)
			JOIN book USING (book_code)
            JOIN wrote USING (book_code)
            JOIN author USING (author_num)
WHERE branch_name = 'JM on the Hill'
	  AND copy_quality = 'Excellent'
ORDER BY book_title, wrote_sequence;

-- Q10 alt using a join-subquery combo
SELECT book_title AS 'title',
	   author_first,
	   author_last
FROM copy JOIN book USING (book_code)
		  JOIN wrote USING (book_code)
          JOIN author USING (author_num)
WHERE copy_quality = 'Excellent'
	  AND branch_num =
      (SELECT branch_num
      FROM branch
      WHERE branch_name = 'JM on the Hill')
ORDER by wrote_sequence;

-- Q11
CREATE TABLE FictionCopies (
    book_code VARCHAR(5),
    book_title VARCHAR(100),
    branch_num INT(2),
    copy_num INT(2),
    copy_quality VARCHAR(10),
    copy_price DECIMAL(5,2)
);

INSERT INTO FictionCopies
SELECT book_code, book_title, branch_num, copy_num, copy_quality, copy_price
FROM book JOIN copy USING (book_code)
WHERE book_type = 'FIC';

SELECT *
FROM fictioncopies;

-- Q12
SELECT book_title,
	   branch_num,
       copy_num,
       copy_quality,
       copy_price AS 'old price',
       if(copy_price < 10.00, round(copy_price*1.1, 2), ' ') AS 'increased price'
FROM fictioncopies
WHERE copy_quality = 'Excellent';

-- Q13
SELECT distinct book_code,
	   book_title AS 'title'
FROM book JOIN copy USING (book_code)
WHERE publisher_code = 
	(SELECT publisher_code
	 FROM publisher
	 WHERE publisher_name = 'Vintage Books')
	OR branch_num =
	   (SELECT branch_num
	    FROM branch
        WHERE branch_name = 'JM Brentwood');

-- Q13 alt using joins
SELECT distinct book_code,
	   book_title AS 'title'
FROM book JOIN publisher USING (publisher_code)
		  JOIN copy USING (book_code)
          JOIN branch USING (branch_num)
WHERE publisher_name = 'Vintage Books'
	  OR branch_name = 'JM Brentwood';

-- Q13 alt using subqueries
SELECT distinct book_code,
	   book_title AS 'title'
FROM book
WHERE publisher_code = 
	(SELECT publisher_code
	 FROM publisher
	 WHERE publisher_name = 'Vintage Books')
	OR book_code IN
		(SELECT book_code
         FROM copy
         WHERE branch_num =
			(SELECT branch_num
             FROM branch
             WHERE branch_name = 'JM Brentwood'));

-- Q14
SELECT book_title AS 'title',
	   publisher_name,
       author_last,
       author_first
FROM book JOIN publisher USING (publisher_code)
		  JOIN wrote USING (book_code)
          JOIN author USING (author_num)
WHERE (SELECT book_code
	   FROM wrote AS wrote2
       WHERE wrote2.book_code = book.book_code
       GROUP BY wrote2.book_code
       HAVING count(*) = 2);

-- Q15
SELECT publisher_name
FROM publisher
WHERE publisher_code NOT IN
	(SELECT publisher_code
     FROM copy JOIN book USING (book_code));

-- Q15 alt (book can exist without a corresponding copy)
SELECT distinct publisher_name
FROM publisher JOIN book USING (publisher_code)
WHERE publisher_code NOT IN
	(SELECT publisher_code
    FROM copy JOIN book USING (book_code));

-- Q16
SELECT branch_name,
	   concat('$', cast(sum(comp_cost) AS CHAR)) AS 'total cost of computers'
FROM branch JOIN employee USING (branch_num)
			JOIN hire USING (emp_num)
            JOIN computer USING (comp_num)
WHERE hire_end IS NULL
GROUP BY branch_name;

-- Q17
SELECT comp_num,
	   year(comp_purchase_date) AS 'year purchased'
FROM computer
WHERE year(comp_purchase_date) <= 2008
	  AND comp_num NOT IN
			(SELECT comp_num
            FROM hire JOIN employee USING (emp_num)
            WHERE (emp_last = 'Chopping' AND emp_first = 'Marie')
				  OR (emp_last = 'Smith' AND emp_first = 'Ronald'));

-- Q17 alt using subqueries
SELECT comp_num,
	   year(comp_purchase_date) AS 'year purchased'
FROM computer
WHERE year(comp_purchase_date) <= 2008
	  AND comp_num NOT IN
		(SELECT comp_num
		FROM hire
        WHERE emp_num IN
			(SELECT emp_num
            FROM employee
            WHERE (emp_last = 'Chopping' AND emp_first = 'Marie')
				  OR (emp_last = 'Smith' AND emp_first = 'Ronald')));

-- Q18
SELECT branch_name,
	   count(emp_num) AS 'num of employees'
FROM branch JOIN employee USING (branch_num)
GROUP BY branch_name
ORDER BY 2 desc;

-- Q19
SELECT distinct book_code,
	   book_title AS 'title'
FROM book JOIN copy USING (book_code)
		  JOIN branch USING (branch_num)
WHERE branch_name = 'JM Brentwood'
	  AND book_code IN
		(SELECT book_code
         FROM book JOIN copy USING (book_code)
				   JOIN branch USING (branch_num)
		 WHERE branch_name = 'JM On the Hill');

-- Q19 alt using one less join         
SELECT distinct book_code,
	   book_title AS 'title'
FROM book JOIN copy USING (book_code)
		  JOIN branch USING (branch_num)
WHERE branch_name = 'JM Brentwood'
	  AND book_code IN
		(SELECT book_code
         FROM copy JOIN branch USING (branch_num)
		 WHERE branch_name = 'JM On the Hill');

-- Q20
SELECT distinct publisher_name
FROM publisher JOIN book USING (publisher_code)
			   JOIN copy USING (book_code)
WHERE branch_num =
	(SELECT max(2)
	FROM (SELECT branch_num, count(distinct book_code)
		  FROM copy JOIN branch USING (branch_num)
		  GROUP BY branch_num) AS sub);