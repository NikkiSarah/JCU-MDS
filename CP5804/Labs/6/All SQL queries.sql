-- Q1 Display the patron id, book number and days kept for every checkout
SELECT pat_id AS 'Patron',
	   book_num AS 'Book',
       datediff(check_in_date, check_out_date) AS 'Days Kept'
FROM checkout;

-- Q2 Display the id, full name and type of every patron
SELECT pat_id,
	   concat(pat_fname, ' ', pat_lname) AS 'Patron Name',
       pat_type
FROM patron;

-- Q3 Display the book number, title with year and subject for every book
SELECT book_num,
	   concat(book_title, ' (', book_year, ')') AS 'Book',
	   book_subject
FROM book;

-- Q4 Display the author's last and first name, and book number for
-- every book written by that author
SELECT au_lname, au_fname, book_num
FROM author
JOIN writes USING (au_id);

-- Q5 Display the author id, book number, title and year for every book
-- *ORDER BY statement included to match the provided output
SELECT au_id, book_num, book_title, book_year
FROM writes JOIN book USING (book_num)
ORDER by book_num;

-- Q6 Display the author's last and first name, book title and year for
-- every book
-- *ORDER BY statement included to match the provided output
SELECT au_lname, au_fname, book_title, book_year
FROM author
JOIN writes USING (au_id)
JOIN book USING (book_num)
ORDER BY book_num;

-- Q7 Display the patron id, book number, patron's first and last name,
-- and book title for every book currently checked out. Sort the results
-- by the patron's last name and book title
SELECT pat_id, book_num, pat_fname, pat_lname, book_title
FROM patron
JOIN book USING (pat_id)
ORDER BY pat_lname, book_title;

-- Q8 Display the patron id, full name and type for every patron. Sort the
-- results by patron type, then last and first name
SELECT pat_id,
	   concat(pat_fname, ' ', pat_lname) AS 'Name',
       pat_type
FROM patron
ORDER BY lower(pat_type), lower(pat_lname), lower(pat_fname);

-- Q9 Display the book number and the number of times each book has been
-- checked out. Do not include books that have never been checked out
-- *ORDER BY statement included to match the provided output
SELECT book_num, count(check_out_date) AS 'Times Checked Out'
FROM checkout
GROUP BY book_num
ORDER BY 2 desc, book_num desc;

-- Q10 Display the author id, first and last name, book number and book
-- title of all books with the subject "Cloud". Sort the results by book
-- title, then the author's last name
SELECT au_id, au_fname, au_lname, book_num, book_title
FROM author JOIN writes USING (au_id)
			JOIN book USING (book_num)
WHERE book_subject = 'Cloud'
ORDER BY book_title, au_lname;

-- Q11 Display the book number, title, author's last and first name,
-- patron id, last name and patron type for every book currently checked
-- out. Sort the results by book title
SELECT book_num, book_title, au_lname, au_fname, pat_id, pat_lname, pat_type
FROM book JOIN patron USING (pat_id)
		  JOIN writes USING (book_num)
		  JOIN author USING (au_id)
ORDER BY book_title;

-- Q12 Display the book number, title and number of times each book has
-- been checked out. Include books that have never been checked out. Sort
-- the results in descending order by the number of times checked out,
-- then by title
SELECT book.book_num, book_title, count(check_num) AS 'Times Checked Out'
FROM book
LEFT JOIN checkout
ON book.book_num = checkout.book_num
GROUP BY book.book_num
ORDER BY 3 desc, book_title;

-- Q13 Display the book number, title and number of times each book has
-- been checked out. Limit the results to books that have been checked out
-- more than five times. Sort the results in descending order by the
-- number of times checked out, then by title
SELECT book.book_num, book_title, count(check_num) AS 'Times Checked Out'
FROM book JOIN checkout USING (book_num)
GROUP BY book.book_num
HAVING count(check_num) > 5
ORDER BY 3 desc, book_title;

-- Q14 Display the author id, author's last name, book title, checkout
-- date, and patron's last name for every book written by an author with
-- last name "Bruer" and have been checked out by a patron with the last
-- name "Miles"
SELECT a.au_id, au_lname, book_title, check_out_date, pat_lname
FROM author AS a,
	 writes AS w,
     book AS b,
     checkout AS c,
     patron AS p
WHERE a.au_id = w.au_id
	  AND w.book_num = b.book_num
      AND b.book_num = c.book_num
      AND c.pat_id = p.pat_id
      AND au_lname = 'Bruer'
      AND pat_lname = 'Miles';
      
-- Q15 Display the patron id, and patron's first and last name of every
-- patron that has never checked out any book. Sort the results by the
-- patron's last name then first name
SELECT patron.pat_id, pat_fname, pat_lname
FROM patron
LEFT JOIN checkout
ON patron.pat_id = checkout.pat_id
WHERE check_num IS NULL
ORDER BY pat_lname, pat_fname;

-- Q16 Display the patron id, last name, number of times that patron has
-- ever checked out a book, and the number of different books the patron
-- has ever checked out. Limit the results to patrons that have made at
-- least 3 checkouts. Sort the results in descending order by number of
-- books, then number of checkouts, then in ascending order by patron id
SELECT pat_id,
	   pat_lname,
	   count(check_num) AS 'Num Checkouts',
       count(distinct book_num) AS 'Num Different Books'
FROM patron JOIN checkout USING (pat_id)
GROUP BY pat_id
HAVING count(check_num) >= 3
ORDER by 4 desc, 3 desc, pat_id;

-- Q17 Display the average number of days a book is kept during a checkout
SELECT round(avg(datediff(check_in_date, check_out_date)), 2) AS 'Average Days Kept'
FROM checkout;

-- Q18 Display the patron id and the average number of days that patron
-- keeps books during a checkout. Limit the results to only patrons that
-- have at least three checkouts. Sort the results in descending order by
-- the average day the book is kept
SELECT pat_id, round(avg(datediff(check_in_date, check_out_date)), 2) AS 'Average Days Kept'
FROM checkout
GROUP BY pat_id
HAVING count(check_num) >= 3
ORDER BY 2 desc;

-- Q19 Display the book number, title and cost of books that have the
-- lowest cost of any books in the system. Sort the results by book number
SELECT book_num, book_title, book_cost
FROM book
WHERE book_cost = (SELECT min(book_cost)
				   FROM book)
ORDER BY book_num;

-- Q20 Display the author id, and author's first and last name for all
-- authors that have never written a book on the subject of
-- 'Programming'. Sort the results by the author's last name
SELECT au_id, au_fname, au_lname
FROM author
WHERE au_id NOT IN (SELECT au_id
					FROM book JOIN writes USING (book_num)
                    WHERE book_subject = 'Programming')
ORDER BY au_lname;

-- Q21 Display the book number, title, subject, average costs of books
-- within that subject, and the difference between each book's cost and
-- the average cost of books in that subject. Sort the results by book
-- title
-- *sorted by book_num instead of book_title to match the provided output
SELECT book_num,
	   book_title,
	   book_subject,
       round(avg_cost, 2) AS 'Avg Cost',
       round(book_cost - avg_cost, 2) AS 'Difference'
FROM book
JOIN (SELECT book_subject,
			 avg(book_cost) as avg_cost
	  FROM book
      GROUP BY book_subject) AS sub USING (book_subject)
ORDER BY book_num;

-- Q22 Display the book number, title, subject, author's last name, and
-- number of books written by that author. Limit the results to books
-- in the 'Cloud' subject. Sort the results by book title, then the
-- author's last name
SELECT book_num,
	   book_title,
       book_subject,
       au_lname,
       num_books AS 'Num Books by Author'
FROM book JOIN writes USING (book_num)
		  JOIN author USING (au_id)
JOIN (SELECT au_lname,
			 count(*) AS num_books
	  FROM author JOIN writes USING (au_id)
      GROUP BY au_lname) AS aut USING (au_lname)
WHERE book_subject = 'Cloud'
ORDER BY book_title, au_lname;

-- Q23 Display the lowest average cost of books within a subject and the
-- highest average cost of books within a subject
SELECT min(avg_cost) AS 'Lowest Avg Cost',
	   max(avg_cost) AS 'Highest Avg Cost'
FROM (SELECT book_subject,
			 round(avg(book_cost), 2) AS avg_cost
	  FROM book
      GROUP BY book_subject) AS subcost;
