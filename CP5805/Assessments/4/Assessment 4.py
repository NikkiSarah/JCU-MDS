#!/usr/bin/env python
# coding: utf-8

# # Asessment 4
# 
# ## Data Analysis - Implement
# 
# ### 13848336 Nikki Fitzherbert

# In[ ]:


import csv

def main():
    user_has_loaded_data = False
    while not user_has_loaded_data:
        try:
            display_main_menu()
            menu_choice = input("Please choose a menu option by entering an integer from 1 to 6: ")

            while menu_choice != "6":
                if menu_choice == "1":
                    filename = get_valid_filename()
                    data = load_data(filename)
                elif menu_choice == "2":
                    display_data(data)
                elif menu_choice == "3":
                    display_set_choice_menu(menu_choice, data)
                    chosen_set = get_set_choice(data)
                    proposed_name = get_new_set_name(data)
                    is_new_set_name_valid(data, proposed_name)
                    rename_set(chosen_set, proposed_name)
                elif menu_choice == "4":
                    display_set_choice_menu(menu_choice, data)
                    chosen_set = get_set_choice(data)
                    sort_set(chosen_set, data)
                elif menu_choice == "5":
                    print("Warning: If you haven't sorted your data first, the median will likely be incorrect.\n")
                    display_set_choice_menu(menu_choice, data)
                    chosen_set = get_set_choice(data)
                    statistics = compile_statistics(chosen_set)
                    display_statistical_report(chosen_set, statistics)
                else:
                    print("You've entered a string or an invalid number. Please try again.\n")

                display_main_menu()
                menu_choice = input("Please choose a menu option by entering an integer from 1 to 6: ")

            print("Goodbye. You have exited the program.")
            return menu_choice

            user_has_loaded_data = True

        except UnboundLocalError:
            print("Have you forgotten to load some data first you numpty?\n")    

def display_main_menu():
    print("Welcome to the Smart Statistician!")
    print("Please choose from the following options:")
    print("  1 – Load data from a file")
    print("  2 – Display the data to the screen")
    print("  3 – Rename a set")
    print("  4 – Sort a set")
    print("  5 – Analyse a set")
    print("  6 - Quit")

def get_valid_filename():
    valid_filename = False
    while not valid_filename:
        try:
            filename = input("Please enter the name of your file including the extension: ")
            open(filename, 'r')
            return filename

            valid_filename = True

        except FileNotFoundError:
            print("That file cannot be found. Please try again.\n")

def load_data(filename):
    file = open(filename, 'r')
    file_reader = csv.reader(file)
    stripped_whitespace_data = [[value for value in row if value != ''] for row in file_reader]
    data = [[int(value) if value.isdigit() else value for value in row] for row in stripped_whitespace_data]
    
    if not any(data):
        print("No data found in that file! Please select option 1 from the main menu and load a different file.\n")
    else:
        print("You have successfully loaded a file with data.\n")
        return data
    
def display_data(data):
    for set in data:
        print(set[0])
        print(set[0:][1:])
        print("---------- \n")

def display_set_choice_menu(menu_choice, data):
    menu_action_option = menu_choice
    number_of_sets = len(data)

    if menu_action_option == "3":
        print("Which set do you want to {}? ".format("rename"))
    elif menu_action_option == "4":
        print("Which set do you want to {}? ".format("sort"))
    else:
        print("Which set do you want to {}? ".format("analyse"))

    for integer in range(1, number_of_sets + 1):
        set_name = data[integer - 1][0]
        print(integer, "-", set_name)
    print()
    
def get_set_choice(data):
    number_of_sets = len(data)
    
    is_integer_input = False
    while not is_integer_input:
        try:
            set_choice = input("Please choose a set by entering a number from {} to {}: ".format(1, number_of_sets))

            if not 1 <= int(set_choice) <= number_of_sets:
                print("That was outside the range of available sets. Please try again.\n")
            else:
                chosen_set = data[int(set_choice) - 1]
                return chosen_set
            
            set_choice = input("Please choose a set by entering a number from {} to {}: ".format(1, number_of_sets))
            
            is_integer_input = True
        
        except ValueError:
            print("That was not an integer. Please try again.\n")
    
    chosen_set = data[int(set_choice) - 1]
    return chosen_set

def get_new_set_name(data):
    proposed_name = input("What would you like to rename the set? ")

    while not is_new_set_name_valid(data, proposed_name):
        print("That name is invalid. A new name for a set must meet the following requirements:")
        print("  1. It cannot be blank.")
        print("  2. It cannot match the name of another set in the file. \n")
        
        proposed_name = input("What would you like to rename the set? ")
    
    return proposed_name

def is_new_set_name_valid(data, proposed_name):
    current_names_list = [element[0] for element in data]
    
    valid_set_name = False
    not_empty_string = False
    doesnt_already_exist = False
    
    for name in proposed_name:
        if proposed_name.strip() != "":
            not_empty_string = True
        if proposed_name not in current_names_list:
            doesnt_already_exist = True
    
    valid_set_name = not_empty_string and doesnt_already_exist
    return valid_set_name

def rename_set(chosen_set, proposed_name):
    current_name = chosen_set[0]
    new_name = proposed_name
    chosen_set[0] = new_name
    print("{} has been renamed to {}.\n".format(current_name, new_name))

def sort_set(chosen_set, data):
    chosen_set.sort(key = lambda value: (isinstance(value, int), value))
    print("You have successfully sorted that set.\n")
    return data
    
def get_set_length(chosen_set):
    number_of_values = len(chosen_set) - 1
    return number_of_values

def get_set_minimum_value(chosen_set):
    minimum_value = min(chosen_set[1:])
    return minimum_value

def get_set_maximum_value(chosen_set):
    maximum_value = max(chosen_set[1:])
    return maximum_value

def get_set_median(number_of_values, chosen_set):
    if number_of_values % 2 == 0:
        median = 0.5 * (chosen_set[number_of_values // 2 - 1] + (chosen_set[number_of_values // 2]))
    else:
        median = chosen_set[number_of_values // 2]
    return median

def get_set_mode(chosen_set):
    max_count = max(map(chosen_set.count, chosen_set))
    all_modes_in_set = list(set(filter(lambda value: chosen_set.count(value) == max_count, chosen_set)))

    if len(all_modes_in_set) > 1:
        mode = "No unique mode"
    else:
        mode = all_modes_in_set[0]
    
    return mode

def compile_statistics(chosen_set):
    number_of_values = get_set_length(chosen_set)
    minimum_value = get_set_minimum_value(chosen_set)
    maximum_value = get_set_maximum_value(chosen_set)
    median = get_set_median(number_of_values, chosen_set)
    mode = get_set_mode(chosen_set)
    
    statistics = [number_of_values, minimum_value, maximum_value, median, mode]
    return statistics

def display_statistical_report(chosen_set, statistics):
    set_name = chosen_set[0]
    
    print(set_name)
    print("----------")
    print("{:<20}".format("Number of values (n)" + ": " + str(statistics[0])))
    print("{:<20}".format("Min" + "{:>19}".format(": ") + "{:<20}".format(str(statistics[1]))))
    print("{:<20}".format("Max" + "{:>19}".format(": ") + "{:<20}".format(str(statistics[2]))))
    print("{:<20}".format("Median" + "{:>16}".format(": ") + "{:<20}".format(str(statistics[3]))))
    print("{:<20}".format("Mode" + "{:>18}".format(": ") + "{:<20}".format(str(statistics[4]))))
    print()
    
main()

