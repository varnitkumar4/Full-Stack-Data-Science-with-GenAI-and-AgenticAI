import tkinter as tk

root = tk.Tk()
root.title('Simple Tkinter App')
root.geometry('200x100')

def say_hello():
    print("Hello, World!")

hello_button = tk.Button(root,text='Click Me',command=say_hello)
hello_button.pack(pady=20)

root.mainloop()