import random

from enum import Enum


class Suit(Enum):
    CLUBS = 1
    DIAMONDS = 2
    HEARTS = 3
    SPADES = 4


class Direction(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class DayOfWeek(Enum):
    SUNDAY = 1
    MONDAY = 2
    TUESDAY = 3
    WEDNESDAY = 4
    THURSDAY = 5
    FRIDAY = 6
    SATURDAY = 7


def print_date(day, month, year):
    print(f"{day}-{month}-{year}")
    DAYS_IN_MONTH = 31
    WEEKS_IN_MONTH = 4
    DAYS_IN_WEEK = 7
    START_YEAR = 2024
    for month in range(1,13):
        print(f"Month : {month}")
        for week in range(1, WEEKS_IN_MONTH + 1):
            print(f"Week : {week}")
            for day in range(1, DAYS_IN_MONTH):
                day_of_week = DayOfWeek((day + month - 1) % DAYS_IN_WEEK + 1)
                print(f"Day : {day}/{month}/{START_YEAR}"
                      f"(Day:{day_of_week},"
                      f"Suit:{Suit},"
                      f"Direction:{Direction})")
                if month == 12 and week == WEEKS_IN_MONTH and day == DAYS_IN_MONTH:
                    START_YEAR = + 1
                    print()

if __name__ == "__main__":
    print_date(2, 10,2024)