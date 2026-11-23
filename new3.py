from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self,colour):
        self.__colour = colour

    @abstractmethod
    def area(self):
        pass

    def get_color(self):
        return self.__colour

    def set_color(self,color):
        self.__colour = color


class Circle(Shape):
    def __init__(self, colour,radius):
        super().__init__(colour)
        self.__radius = radius
    def area(self):
            return 3.14 * (self.__radius ** 2)  # Circle area formula

    # Getter for radius
    def get_radius(self):
        return self.__radius

    # Setter for radius
    def set_radius(self, radius):
        self.__radius = radius

# Rectangle class that inherits from Shape
class Rectangle(Shape):
    def __init__(self, color, width, height):
        super().__init__(color)
        self.__width = width  # Private attribute (Encapsulation)
        self.__height = height  # Private attribute (Encapsulation)

    def area(self):
        return self.__width * self.__height  # Rectangle area formula

    # Getter for width
    def get_width(self):
        return self.__width

    # Setter for width
    def set_width(self, width):
        self.__width = width

    # Getter for height
    def get_height(self):
        return self.__height

    # Setter for height
    def set_height(self, height):
        self.__height = height

# Demonstrating Polymorphism
def print_area(shape):
    print(f"The area of the {shape.__class__.__name__} is: {shape.area()} square units")

# Creating objects
circle = Circle("Red", 5)
rectangle = Rectangle("Blue", 4, 6)

# Using polymorphism to calculate the area of different shapes
print_area(circle)  # Polymorphism: Will call Circle's area method
print_area(rectangle)  # Polymorphism: Will call Rectangle's area method

# Encapsulation: Accessing and modifying private variables through getters and setters
print(f"Circle color: {circle.get_color()}")
circle.set_color("Green")
print(f"Updated circle color: {circle.get_color()}")

# Modifying attributes
rectangle.set_width(5)
rectangle.set_height(8)
print(f"Updated rectangle area: {rectangle.area()}")