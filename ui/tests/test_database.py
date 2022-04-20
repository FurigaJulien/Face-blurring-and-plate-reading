import pytest
from sqlmodel import Session, SQLModel, create_engine,select,engine
from sqlmodel.pool import StaticPool
from app.Database import Users,Plates

@pytest.fixture(name="engine")
def session_fixture():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    SQLModel.metadata.create_all(engine)
    yield engine



def test_insert_and_read_user(engine: engine):
    with Session(engine) as session:
        user = Users(username="test", password="test", first_name="test", family_name="test")
        session.add(user)
        session.commit()

    result = Users.get_user(engine, "test")

    assert result.username == "test"
    assert result.password == "test"
    assert result.first_name == "test"
    assert result.family_name == "test"

def test_insert_user_and_check_password(engine:engine):
    with Session(engine) as session:
        user = Users(username="test", password="test", first_name="test", family_name="test")
        session.add(user)
        session.commit()

    assert Users.check_password(engine, "test", "test") == (True, 1)
    assert Users.check_password(engine, "test", "test2") == (False, '')

def test_get_username_availability(engine:engine):
    with Session(engine) as session:
        user = Users(username="test", password="test", first_name="test", family_name="test")
        session.add(user)
        session.commit()

    assert Users.get_username_availability(engine, "test") == False
    assert Users.get_username_availability(engine, "test2") == True


def test_insert_plate(engine:engine):
    with Session(engine) as session:
        user = Users(username="test", password="test", first_name="test", family_name="test")
        session.add(user)
        session.commit()

    with Session(engine) as session:
        plate = Plates(user_id=1, plate_number="test", plate_type="test")
        session.add(plate)
        session.commit()

    with Session(engine) as session:
        statement = select(Plates).where(Plates.plate_number == "test")
        result = session.exec(statement).all()

    assert len(result) == 1
    assert result[0].user_id == '1'
    assert result[0].plate_number == "test"


def test_get_plates_for_user(engine:engine):
    with Session(engine) as session:
        user = Users(username="test", password="test", first_name="test", family_name="test")
        session.add(user)
        session.commit()

    with Session(engine) as session:
        plate = Plates(user_id=1, plate_number="test", plate_type="test")
        session.add(plate)
        session.commit()

    result = Plates.get_plates_for_user(engine, 1)

    assert len(result) == 1
    assert result[0].user_id == '1'
    assert result[0].plate_number == "test"
