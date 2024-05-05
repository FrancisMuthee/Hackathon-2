class Dog: # Name class, Capital letter

    def __init__(self, name) -> None:
        self.name = name
        print(name)

    def add_one(self, x):
        return x+1


    def bark(self):
        print("bark") # operation 

d = Dog("Tim")
print(d.name)
d2 = Dog('Bill')
d.bark()
print(type(d))
print(d.add_one(5))