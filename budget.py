import json
import os

class BudgetTracker:
    def __init__(self, filename='expenses.json'):
        self.filename = filename
        self.expenses = self.load_expenses()
        self.budget = self.load_budget()

    def load_expenses(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        else:
            return []

    def save_expenses(self):
        with open(self.filename, 'w') as f:
            json.dump(self.expenses, f)

    def load_budget(self):
        if os.path.exists('budget.txt'):
            with open('budget.txt', 'r') as f:
                return float(f.read())
        else:
            return None

    def save_budget(self):
        with open('budget.txt', 'w') as f:
            f.write(str(self.budget))

    def view_budget(self):
        if self.budget is None:
            print("No budget set. Please set a budget using the 'set_budget' command.")
        else:
            print(f"Your current budget is: ${self.budget:.2f}")

    def set_budget(self):
        while True:
            try:
                self.budget = float(input("Enter your budget: $"))
                self.save_budget()
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    def add_expense(self):
        name = input("Enter expense name: ")
        while True:
            try:
                amount = float(input("Enter expense amount: $"))
                self.expenses.append({'name': name, 'amount': amount})
                self.save_expenses()
                self.check_budget_alert()
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    def view_expenses(self):
        if not self.expenses:
            print("No expenses recorded.")
        else:
            print("Your expenses:")
            for i, expense in enumerate(self.expenses, start=1):
                print(f"{i}. {expense['name']}: ${expense['amount']:.2f}")

    def delete_expense(self):
        if not self.expenses:
            print("No expenses recorded.")
        else:
            self.view_expenses()
            while True:
                try:
                    choice = int(input("Enter the number of the expense to delete: "))
                    if 1 <= choice <= len(self.expenses):
                        del self.expenses[choice - 1]
                        self.save_expenses()
                        break
                    else:
                        print("Invalid choice. Please enter a number between 1 and", len(self.expenses))
                except ValueError:
                    print("Invalid input. Please enter a number.")

    def check_budget_alert(self):
        if self.budget is not None:
            total_expenses = sum(expense['amount'] for expense in self.expenses)
            if total_expenses / self.budget >= 0.95:
                print("Budget alert: You have used 95% or more of your budget.")

def main():
    tracker = BudgetTracker()
    while True:
        print("\n1. View budget")
        print("2. Set budget")
        print("3. Add expense")
        print("4. View expenses")
        print("5. Delete expense")
        print("6. Quit")
        choice = input("Enter your choice: ")
        if choice == "1":
            tracker.view_budget()
        elif choice == "2":
            tracker.set_budget()
        elif choice == "3":
            tracker.add_expense()
        elif choice == "4":
            tracker.view_expenses()
        elif choice == "5":
            tracker.delete_expense()
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()