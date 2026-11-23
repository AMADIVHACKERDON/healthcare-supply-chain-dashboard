class Engine:
    def __init__(self,engine_type,horsepower ):
        self.engine_type = engine_type
        self.horsepower = horsepower

    def start(self):
        print(f"starting way {self.engine_type} with {self.horsepower} ")
    
    def stop(self):
        print(f"stopping with {self.engine_type} a {self.horsepower}")

class Vehicle:
    def __init__(self, model, make, year):
        self.make = make
        self.model = model
        self.year = year

    def start(self):
        print(f"the model {self.make} {self.model} {self.year} is starting")
    def stop(self):
        print(f"the model {self.make} {self.model} {self.year} is stoping")

    def displayinfo(self):
        print(f"display the {self.make} {self.model} {self.year}")

class Car(Vehicle):
    def __init__(self, make, model, year,engine ):        
        super().__init__(make, model, year)
        self.engine = engine

    def start(self):
        super().start()
        self.engine.start()
    def stop(self):
        super().stop()
        self.engine.stop()


class Truck(Vehicle):
    def __init__(self, model, make, year,engine, cargo_capacity):
        super().__init__(model, make, year)
        self.engine = engine
        self .cargo_capacity = cargo_capacity
    
    def start(self):
        super().start()
        self.engine.start()
    def stop(self):
        super().stop()
        self.engine.stop()

    def loadcargo(self,weight):
        if weight <= self .cargo_capacity:
            print(f"loading {weight}kg")

car_engine = Engine("V6","300") 
truck_engine = Engine("V8","400")

car = Car("Toyata", "Camry","2024", car_engine)
print(car.displayinfo())
print(car.start())

truck = Truck("CR-V","Honda","2023",truck_engine,900)
print(truck.loadcargo(700))