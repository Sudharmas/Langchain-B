from typing import Optional

from pydantic import BaseModel,Field,EmailStr

class student(BaseModel):
    name: str = 'xyz'
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10)
    email : EmailStr


newstudent = {'age': 18, 'cgpa': '7.6', 'email': 'qwe@abc.com'}
student = student(**newstudent)
print(student)