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
        engine : SQLModel.engine
        username : str
            User name

        Returns
        -------
        Optional[Users]
            return a user if it exists
        """
        with Session(engine) as session:
            statement = select(Users).where(col(Users.username)==username)
            results = session.exec(statement).all()
            if len(results)>0:
                return results[0]
            else:
                return None

    @classmethod
    def get_username_availability(cls,engine,username)->bool:
        """Function to check that a username is not already taken in the database

        Parameters
        ----------
        engine : _type_
            _description_
        username : _type_
            _description_

        Returns
        -------
        bool
            boolean whether the username is available or not
        """
        if cls.get_user(engine,username) == None:
            return True
        else:
            return False

class Plates(SQLModel, table=True):
    """SQL Model plates class

    Parameters
    ----------
    SQLModel : SQLModel
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
        engine : SQLModel.engine
        plate : Plate
            plate object defined by the user
        """
        with Session(engine) as session:
            session.add(plate)
            session.commit()

    @classmethod
    def get_plates_for_user(cls,engine,user_id:int):
        """Get all plates already detected for a user

        Parameters
        ----------
        engine : SQLModel.engine
        user_id : int
            user id

        Returns
        -------
        Optional[Plates]
            all plates detected for a user
        """
        with Session(engine) as session:
            statement = select(Plates).where(col(Plates.user_id)==user_id)
            results = session.exec(statement).all()
            if len(results)>0:
                print(results)
                return results
            else:
                return None


class Files(SQLModel, table=True):
    """_summary_

    Parameters
    ----------
    SQLModel : SQLModel
    table : bool, optional
        _description_, by default True
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    file_name: str


    @classmethod
    def insert_file(cls,engine,file):
        """Insert file in database

        Parameters
        ----------
        engine : SQL Model.engine
        file : File
            file object defined by the user
        """
        with Session(engine) as session:
            session.add(file)
            session.commit()

    @classmethod
    def get_files_for_user(cls,engine,user_id:int):
        """Get all files already detected for a user

        Parameters
        ----------
        engine : SQLModel.engine
        user_id : int
            user id

        Returns
        -------
        Optional[Files]
            return all files detected for a user
        """
        with Session(engine) as session:
            statement = select(Files).where(col(Files.user_id)==user_id)
            results = session.exec(statement).all()
            if len(results)>0:
                return results
            else:
                return None

