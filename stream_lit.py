import streamlit as st

def main():
    st.title("User List App")

    # Sample user data
    users = [
        {"id": 1, "name": "John Doe", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
        {"id": 3, "name": "Bob Johnson", "email": "bob@example.com"},
    ]

    # Display the list of users
    selected_user = show_user_list(users)

    # Display modal when a user is selected
    if selected_user:
        show_user_modal(selected_user)

def find_data_with_list_comprehension(data_list, key, value):
    matches = [item for item in data_list if item.get(key) == value]
    return matches[0] if matches else {}

def show_user_list(users):
    st.header("User List")

    # Display a list of user names as buttons
    user_names = [user["name"] for user in users]
    selected_user_index = st.selectbox("Select a user:", user_names)

    # Return the selected user based on the index
    return find_data_with_list_comprehension(users, "name", selected_user_index) if users else None

def show_user_modal(user):
    st.success(f"User Information: {user['name']}")
    st.write(f"User ID: {user['id']}")
    st.write(f"Email: {user['email']}")

if __name__ == "__main__":
    main()
