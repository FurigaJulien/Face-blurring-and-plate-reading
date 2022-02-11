import string
from tokenize import String
from typing import Optional
import os
from unittest import result
from xmlrpc.client import Boolean
from sqlmodel import Field, SQLModel, Session, select, col


class Users(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    password: str
    first_name: str
    family_name:str

    @classmethod
    def insert_user(cls,engine,user):

        with Session(engine) as session:
            session.add(user)
            session.commit()

    @classmethod
    def check_password(csl,engine,username:string, password:string):
        """
        Function to check a user password
        
        Parameters
        ----------
        engine : SQLModel.engine
        username : String

        Returns
        -------
        Boolean whether the password is correct or not
        """
        validate = False
        with Session(engine) as session:
            statement = select(Users).where(col(Users.username)==username)
            results = session.exec(statement)

            for user in results:
                if user.password == password:
                    validate = True

        return validate

    @classmethod
    def get_user(cls,engine,username:string):
        with Session(engine) as session:
            statement = select(Users).where(col(Users.username)==username)
            results = session.exec(statement).all()
            if len(results)>0:
                return results[0]
            else:
                return None

    @classmethod
    def get_username_availability(cls,engine,username):
        if cls.get_user(engine,username) == None:
            return True
        else:
            return False

