from typing import TypedDict

class person(TypedDict):
    name: str
    age: int

newperson : person = {'name': 'xyz', 'age': '18'}
print(newperson)
