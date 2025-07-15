from app import app, db

with app.app_context():
    print("Dropping all tables...")
    db.drop_all()
    print("Creating new tables...")
    db.create_all()
    print("Database reset complete!")
