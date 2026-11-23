class Person:
    def __init__(self, feel, food, **kwargs):
        super().__init__(**kwargs)
        self.feel = feel
        self.food = food

    def feeling(self):
        print(f"how are you? {self.feel} with eating {self.food}")

class Animal:
    def __init__(self, specie,name):
        self.specie = specie
        self.name = name
        
    def speak(self):
        print(f"{self.name} is a {self.specie}")

class Dog(Person,Animal):
    def __init__(self, specie, name,feel,food, bronk):
        super().__init__(specie = specie, name = name ,feel = feel,food=feel)
        self.bronk = bronk
    def info(self):
        print("Dog bark")

dog1 = Dog('mammal','bingo','calm','sheet','shelf')
dog1.info()
dog1.speak()
dog1.feeling()