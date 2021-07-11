-- Q1 Display the first and last name of every patron
SELECT pat_fname, pat_lname
FROM patron;

-- Q2 Display the checkout number, date and due date of every book that
-- has been checked out
SELECT check_num, check_out_date, check_due_date
FROM checkout;

-- Q3 Display the book number, book title and publication year of every
-- book
SELECT book_num, book_title AS 'title', book_year AS 'year published'
FROM book;

-- Q4 Display the book number, title and cost of every book
SELECT book_num, book_title, book_cost
FROM book;

-- Q5 Display the different years that books have been published in
SELECT distinct book_year
FROM book;

-- Q6 Display the checkout number, book number, patron id, checkout
-- date and due date for every checkout that has ever occurred. Sort
-- the results in descending order
SELECT check_num, book_num, pat_id, check_out_date, check_due_date
FROM checkout
ORDER BY check_out_date desc;

-- Q7 Display the book title, year and subject for every book. Sort the
-- results by ascending book subject, descending year and ascending
-- title
SELECT book_title, book_year, book_subject
FROM book
ORDER BY book_subject, book_year desc, book_title;

-- Q8 Display the book number, title and publication year for every book
-- published in 2012
SELECT book_num, book_title, book_year
FROM book
WHERE book_year = 2012;

-- Q9 Display the book number, title and publication year for every
-- "Database" book
SELECT book_num, book_title, book_year
FROM book
WHERE book_subject = 'Database';

-- Q10 Display the checkout number, book number and checkout date for
-- all books checked out before 5 April 2015
SELECT check_num, book_num, check_out_date
FROM checkout
WHERE check_out_date < '2015-04-05';

-- Q11 Display the book number, title and publication year for every
-- "Programming" book published after 2013
SELECT book_num, book_title, book_year
FROM book
WHERE book_subject = 'Programming'
	AND book_year > 2013;

-- Q12 Display the book number, title, publication year, subject and
-- cost of every book on "Middleware" or "Cloud" and cost more than $70
SELECT book_num, book_title, book_year, book_subject, book_cost
FROM book
WHERE book_subject IN ('Middleware', 'Cloud')
	AND book_cost > 70;

-- Q13 Display the author id, first name, last name and birth year for
-- every author born in the 1980s
SELECT *
FROM author
WHERE au_birthyear LIKE '198_';

-- Q14 Display the book number, title and publication book for every book
-- containing "Database" in their title irrespective of how it is
-- capitalised
SELECT book_num, book_title, book_year
FROM book
WHERE lower(book_title) LIKE '%database%';

-- Q15 Display the patron id, and first and last name of all patrons that
-- are students
SELECT pat_id, pat_fname, pat_lname
FROM patron
WHERE lower(pat_type) = 'student';

-- Q16 Display the patron id, first and last name, and type of all
-- patrons whose last name begins with the letter "C"
SELECT *
FROM patron
WHERE lower(pat_lname) LIKE 'c%';

-- Q17 Display the author id, and first and last name of all authors
-- whose birth year is unknown
SELECT au_id, au_fname, au_lname
FROM author
WHERE au_birthyear IS NULL;

-- Q18 Display the author id, and first and last name of all authors
-- whose birth year is known
SELECT au_id, au_fname, au_lname
FROM author
WHERE au_birthyear IS NOT NULL;

-- Q19 Display the checkout number, book number, patron id, checkout
-- date and due dates for all checkouts that have not yet been returned.
-- Sort the results by ascending book number
SELECT check_num, book_num, pat_id, check_out_date, check_due_date
FROM checkout
WHERE check_in_date IS NULL
ORDER BY book_num;

-- Q20 Display the author id, first and last name, and birth year of
-- every author. Sort the results by descending birth year, then
-- ascending last name
SELECT au_id, au_fname, au_lname, au_birthyear
FROM author
ORDER BY au_birthyear desc, au_lname;

-- Q21 Display the number of books
SELECT count(*) AS 'Number of Books'
FROM book;

-- Q22 Display the number of different book subjects
SELECT count(distinct book_subject) AS 'Number of Subjects'
FROM book;

-- Q23 Display the number of available books (not currently checked
-- out)
SELECT count(distinct book_num) AS 'Available Books'
FROM checkout
WHERE check_in_date IS NOT NULL;

-- actual answer
SELECT count(book_num) AS 'Available Books'
FROM book
WHERE pat_id IS NULL;

-- Q24 Display the price of the most expensive book
SELECT max(book_cost) AS 'Most Expensive'
FROM book;

-- Q25 Display the price of the cheapest book
SELECT min(book_cost) AS 'Least Expensive'
FROM book;

-- Q26 Display the number of different patrons who have ever checked out
-- a book
SELECT count(distinct pat_id) AS 'Different Patrons'
FROM checkout;

-- Q27 Display the subject and number of books in each subject. Sort
-- the results by the number of books in descending order, then
-- ascending subject name
SELECT book_subject, count(*) AS 'Books in Subject'
FROM book
GROUP BY book_subject
ORDER BY 2 desc, book_subject;

-- Q28 Display the author id and the number of books written by that
-- author. Sort the results by descending number of books, then
-- ascending author id
SELECT author.au_id, count(*) AS 'Books Written'
FROM author, writes
WHERE author.au_id = writes.au_id
GROUP BY author.au_id
ORDER BY 2 desc, author.au_id;

-- Q29 Display the total value of all books in the library
SELECT round(sum(book_cost), 0) AS 'Library Value'
FROM book;