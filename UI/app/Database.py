from typing import Optional
from sqlmodel import Field, SQLModel, Session, select, col


class Users(SQLModel, table=True):
    """SQLModel Users class

    Parameters
    ----------
    SQLModel : _type_
        _description_
    table : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
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
    def check_password(csl,engine,username:str, password:str):
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
        placeholder = ""
        with Session(engine) as session:
            statement = select(Users).where(col(Users.username)==username)
            results = session.exec(statement)

            for user in results:
                if user.password == password:
                    validate = True
                    placeholder = user.id

        return validate,placeholder

    @classmethod
    def get_user(cls,engine,username:str):
        """Get a user in database

        Parameters
        ----------
        engine : _type_
            _description_
        username : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        with Session(engine) as session:
            statement = select(Users).where(col(Users.username)==username)
            results = session.exec(statement).all()
            if len(results)>0:
                return results[0]
            else:
                return None

    @classmethod
    def get_username_availability(cls,engine,username):
        """Function to check that a username is not already taken in the database

        Parameters
        ----------
        engine : _type_
            _description_
        username : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if cls.get_user(engine,username) == None:
            return True
        else:
            return False

class Plates(SQLModel, table=True):
    """SQL Model plates class

    Parameters
    ----------
    SQLModel : _type_
        _description_
    table : bool, optional
        _description_, by default True
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    plate_number: str


    @classmethod
    def insert_plates(cls,engine,plate):
        """Insert plate in database

        Parameters
        ----------
        engine : _type_
            _description_
        user : _type_
            _description_
        """
        with Session(engine) as session:
            session.add(plate)
            session.commit()

    @classmethod
    def get_plates_for_user(cls,engine,user_id:int):
        """Get all plates already detected for a user

        Parameters
        ----------
        engine : _type_
            _description_
        user_id : int
            _description_

        Returns
        -------
        _type_
            _description_
        """
        with Session(engine) as session:
            statement = select(Plates).where(col(Plates.user_id)==user_id)
            results = session.exec(statement).all()
            if len(results)>0:
                print(results)
                return results
            else:
                return None