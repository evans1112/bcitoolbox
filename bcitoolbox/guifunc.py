#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import random
import os
import tkinter as tk
import pandas as pd
import webbrowser
import requests
from scipy.fftpack import dct
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from .BCIbox import simulateVV
from .BCIbox import prod_gaus
from .BCIbox import prod3gauss
from .BCIbox import plotKonrads

from .BCIboxGUI import simulateVV_GUI
from .BCIboxGUI import simulateLC_GUI
from .BCIboxGUI import simulateLCD_GUI

from tkinter import filedialog
from tkinter import PhotoImage
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.stats import norm
from pyvbmc import VBMC
from PIL import Image, ImageTk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def gui():
    global window, data, a, b, c, result_label, Status_label, progressbar, count, bounds, bounds_l, fixvalue, fixvalue_l, checkbox_vars
    global randseeds, randseed_mat, init_P, init_V, init_nor, init_numt

    package_name = 'bcitoolbox'
    expected_version = '0.0.2.6'
    #download resources
    BCI_url = "https://gitee.com/haochengzhu/bcitoolbox/raw/master/resources/BCI.png"
    inst1D_url = "https://gitee.com/haochengzhu/bcitoolbox/raw/master/resources/inst1D.png"
    inst2D_url = "https://gitee.com/haochengzhu/bcitoolbox/raw/master/resources/inst2D.png"
    instdisc_url = "https://gitee.com/haochengzhu/bcitoolbox/raw/master/resources/instdisc.png"
    demo_url = "https://gitee.com/haochengzhu/bcitoolbox/raw/master/resources/demo.csv"

    folder_path = "resources"

    local_BCI = os.path.join(folder_path, "BCI.png")
    local_inst1D = os.path.join(folder_path, "inst1D.png")
    local_inst2D = os.path.join(folder_path, "inst2D.png")
    local_instdisc = os.path.join(folder_path, "instdisc.png")
    local_demo = os.path.join(folder_path, "demo.csv")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if not os.path.exists(local_BCI):
        response1 = requests.get(BCI_url)
        response2 = requests.get(inst1D_url)
        response3 = requests.get(inst2D_url)
        response4 = requests.get(instdisc_url)
        response5 = requests.get(demo_url)
   
        if response1.status_code == 200:
    
            with open(local_BCI, 'wb') as file:
                file.write(response1.content)

            with open(local_inst1D, 'wb') as file:
                file.write(response2.content)
            
            with open(local_inst2D, 'wb') as file:
                file.write(response3.content)
            
            with open(local_instdisc, 'wb') as file:
                file.write(response4.content)

            with open(local_demo, 'wb') as file:
                file.write(response5.content)

    data = None
    a = None
    b = None
    c = None
    result_label = None
    Status_label = None
    progressbar = None
    count = 5
    bounds = [(0, 1),(0.1, 3),(0.1, 3),(0.1, 3),(0, 3.5)]
    bounds_l = [(0, 1),(0.1, 15),(0.1, 15),(5, 100),(-30, 30)]
    fixvalue = [0.5,0.4,0.8,4000,2,70000.5,70000.5]
    fixvalue_l = [0.5,2,8,4000,0,70000.5,70000.5]
    checkbox_vars = np.array([1,1,1,1,1,0,0])
    randseeds = 1
    randseed_mat = [13, 17, 9, 3, 600, 817, 190, 268, 846, 525, 458, 273, 877, 340, 360, 967, 509, 213, 636, 21, 358, 316, 382, 325, 633, 745, 30, 307, 748, 867, 524, 619, 630, 644, 626, 560, 615, 835, 674, 945, 880, 126, 776, 586, 232, 463, 1000, 565, 869, 761, 213, 587, 680, 222, 218, 273, 974, 275, 949, 792, 670, 260, 466, 30, 313, 4, 448, 512, 660, 412, 289, 921, 949, 587, 219, 747, 257, 23, 382, 418, 535, 235, 999, 923, 694, 488, 664, 577, 223, 288, 829, 843, 634, 508, 154, 267, 90, 18, 278, 559]
    init_P = True
    init_V = False

    init_nor = False
    init_numt = True
    # Create Homepage
    window = tk.Tk()
    window.title("Bayesian Toolbox")  # Set name

    # Get screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Set Window width and height
    window_width = 600
    window_height = 700

    # Compute locations
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    # 
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    open_welcome()

    latest_version = get_latest_package_version(package_name)

    # If latest version is fetched successfully and it's not the expected version, show the update popup
    if latest_version and latest_version != expected_version:
        show_update_popup(package_name, expected_version, latest_version)
    # Run
    window.mainloop()

    



def get_latest_package_version(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['info']['version']
    else:
        return None

def show_update_popup(package_name, current_version, latest_version):
    # Create a new tkinter window
    root = tk.Tk()
    root.title("Update Required")

    # Set the window size
    root.geometry("400x200")

    # Add a Text widget with the update instructions
    text_widget = tk.Text(root, height=7, width=50)
    text_widget.pack(pady=10)
    update_message = (f"Your package version ({current_version}) is not up to date.\n"
                      f"The latest version is {latest_version}.\n\n"
                      "Please update using:\n\n")
    text_widget.insert(tk.END, update_message)
    text_widget.insert(tk.END, "pip", 'pip_tag')
    text_widget.insert(tk.END, " install --upgrade ")
    text_widget.insert(tk.END, package_name, 'package_name_tag')
    
    # Configure tags to change the color of 'pip' and 'PACKAGE_NAME'
    text_widget.tag_config('pip_tag', foreground='dark blue')
    text_widget.tag_config('package_name_tag', foreground='dark red')

    # Set a lighter background for the entire text
    text_widget.configure(bg='#f0f0f0')

    # Disable the Text widget to prevent user editing
    text_widget.configure(state='disabled')


    # Run the tkinter event loop
    root.mainloop()


    
def import_csv(text_entry):
    global data_matrices
    global file_paths
    file_paths = text_entry.get()
    if not file_paths:
        file_paths = filedialog.askopenfilenames(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
        text_entry.delete(0, tk.END)
        text_entry.insert(tk.END, file_paths)

    if file_paths:
        data_matrices = []
        for file_path in file_paths:
            print(file_path)
            df = np.loadtxt(file_path, delimiter=',')
            data_matrices.append(df)
            
def import_csv2(text_entry):
    global data_matrices
    global file_paths
    file_paths = text_entry.get()
    if not file_paths:
        file_paths = filedialog.askopenfilenames(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
        text_entry.delete(0, tk.END)
        text_entry.insert(tk.END, file_paths)

    if file_paths:
        data_matrices = []
        for file_path in file_paths:
            print(file_path)
            df = pd.read_csv(file_path, header=None, na_values='NaN')
            df=np.array(df)
            data_matrices.append(df)           
        
def open_file(text_entry):
    global data_matrices
    global file_paths
    file_paths = text_entry.get()
    file_paths = filedialog.askopenfilenames(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
    text_entry.delete(0, tk.END)
    text_entry.insert(tk.END, file_paths)
    if file_paths:
        data_matrices = []
        for file_path in file_paths:
            print(file_path)
            df = np.loadtxt(file_path, delimiter=',')
            data_matrices.append(df)

def demo():
    global entry
    global data_matrices
    global file_paths
    global var_mll, var_mr2, var_sse,var_a,var_b ,var_c
    file_paths='resources/demo.csv'
    entry.delete(0, tk.END)
    entry.insert(tk.END, file_paths)
    if True:
        data_matrices = []
        print(file_paths)
        df = np.loadtxt(file_paths, delimiter=',')
        data_matrices.append(df)

    var_mll = tk.BooleanVar(value=True)
    var_mr2 = tk.BooleanVar()
    var_sse = tk.BooleanVar()

    var_a = tk.BooleanVar(value=True)
    var_b = tk.BooleanVar()
    var_c = tk.BooleanVar()

    run_code()

    


def open_file2(text_entry):
    global data_matrices
    global file_paths
    file_paths = text_entry.get()
    file_paths = filedialog.askopenfilenames(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
    text_entry.delete(0, tk.END)
    text_entry.insert(tk.END, file_paths)
    if file_paths:
        data_matrices = []
        for file_path in file_paths:
            print(file_path)
            df = pd.read_csv(file_path, header=None, na_values='NaN')
            df= np.array(df)
            data_matrices.append(df)          

def open_document():
    file_path = filedialog.askopenfilename()
    if file_path:  
        print(file_path)

def about_bci():

    about_window = tk.Toplevel(welcome_frame)
    about_window.title("About BCI Toolbox")
    
    g_font = ("Georgia", 18, "bold")
    title_font = ("Georgia", 17, "bold")
    name_font = ("Georgia", 15, "bold")
    info_font = ("Script MT Bold", 13)

    title_label = tk.Label(about_window, text="About BCI Toolbox", font=g_font)
    title_label.pack(pady=10)
    names_text = tk.Text(about_window, wrap=tk.WORD, font=name_font, padx=10, pady=10)
    names_text.pack()
    names_info_c = [
        ("Dr. Ladan Shams", "University of California, Los Angeles", "ladan@psych.ucla.edu"),
        ("Dr. Ulrik R. Beierholm", "Durham University", "beierh@gmail.com")   
    ]
    names_info_i = [
        ("Haocheng (Evans) Zhu", "Soochow University", "evanszhu2001@gmail.com"),   
    ]
    names_text.insert(tk.END, "Conceptualization\n\n", "title")
    for name, univ, emailA in names_info_c:
        names_text.insert(tk.END, f"{name}\n", "name")
        names_text.insert(tk.END, f"{univ}\n{emailA}\n\n", "info")

    names_text.insert(tk.END, "\nImplementation\n\n", "title")
    for name, univ, emailA in names_info_i:
        names_text.insert(tk.END, f"{name}\n", "name")
        names_text.insert(tk.END, f"{univ}\n{emailA}\n\n", "info") 

    names_text.tag_configure("title", font=title_font, justify=tk.CENTER)
    names_text.tag_configure("name", font=name_font, justify=tk.CENTER)
    names_text.tag_configure("info", font=info_font, justify=tk.CENTER)

def done_setting():
    global count, checkbox_vars, fixvalue, fixvalue_l
    global bounds, bounds_l, bounds_use, bounds_l_use
    checkbox_vars = [var1.get(), var2.get(), var3.get(), var4.get(), var5.get(), var6.get(), var7.get()]
    checkbox_vars = np.array(checkbox_vars)
    print(checkbox_vars)
    count = sum(checkbox_vars)
    print("Number of free parameters:", count)
    if ifloc == 1:
        bounds_l = [
        [entry1_l.get(), entry1_u.get()],
        [entry2_l.get(), entry2_u.get()],
        [entry3_l.get(), entry3_u.get()],
        [entry4_l.get(), entry4_u.get()],
        [entry5_l.get(), entry5_u.get()],
        [entry6_l.get(), entry6_u.get()],
        [entry7_l.get(), entry7_u.get()]
        ]

        bounds_l= [[float(lower), float(upper)] for lower, upper in bounds_l]
        bounds_l = np.array(bounds_l)
        bounds_l_use = bounds_l[checkbox_vars==1]
        fixvalue_l = [entry1_f.get(), entry2_f.get(), entry3_f.get(), entry4_f.get(), entry5_f.get(), entry6_f.get(), entry7_f.get()]
        fixvalue_l = [float(value) for value in fixvalue_l]
    else :
        bounds = [
        [entry1_l.get(), entry1_u.get()],
        [entry2_l.get(), entry2_u.get()],
        [entry3_l.get(), entry3_u.get()],
        [entry4_l.get(), entry4_u.get()],
        [entry5_l.get(), entry5_u.get()],
        [entry6_l.get(), entry6_u.get()],
        [entry7_l.get(), entry7_u.get()]
        ]
    # 
        bounds= [[float(lower), float(upper)] for lower, upper in bounds]
        bounds = np.array(bounds)
        bounds_use = bounds[checkbox_vars==1]
        fixvalue = [entry1_f.get(), entry2_f.get(), entry3_f.get(), entry4_f.get(), entry5_f.get(), entry6_f.get(), entry7_f.get()]
        fixvalue = [float(value) for value in fixvalue]


    parameters_window.destroy()
    
def save_setting():
    global count, checkbox_vars, fixvalue, fixvalue_l
    global bounds, bounds_l, bounds_use, bounds_l_use
    checkbox_vars = [var1.get(), var2.get(), var3.get(), var4.get(), var5.get(), var6.get(), var7.get()]
    checkbox_vars = np.array(checkbox_vars)
    print(checkbox_vars)
    count = sum(checkbox_vars)
    print("Number of free parameters:", count)
    if ifloc == 1:
        bounds_l = [
        [entry1_l.get(), entry1_u.get()],
        [entry2_l.get(), entry2_u.get()],
        [entry3_l.get(), entry3_u.get()],
        [entry4_l.get(), entry4_u.get()],
        [entry5_l.get(), entry5_u.get()],
        [entry6_l.get(), entry6_u.get()],
        [entry7_l.get(), entry7_u.get()]
        ]

        bounds_l= [[float(lower), float(upper)] for lower, upper in bounds_l]
        bounds_l = np.array(bounds_l)
        bounds_l_use = bounds_l[checkbox_vars==1]
        fixvalue_l = [entry1_f.get(), entry2_f.get(), entry3_f.get(), entry4_f.get(), entry5_f.get(), entry6_f.get(), entry7_f.get()]
        fixvalue_l = [float(value) for value in fixvalue_l]
    else :
        bounds = [
        [entry1_l.get(), entry1_u.get()],
        [entry2_l.get(), entry2_u.get()],
        [entry3_l.get(), entry3_u.get()],
        [entry4_l.get(), entry4_u.get()],
        [entry5_l.get(), entry5_u.get()],
        [entry6_l.get(), entry6_u.get()],
        [entry7_l.get(), entry7_u.get()]
        ]
        bounds= [[float(lower), float(upper)] for lower, upper in bounds]
        bounds = np.array(bounds)
        bounds_use = bounds[checkbox_vars==1]
        fixvalue = [entry1_f.get(), entry2_f.get(), entry3_f.get(), entry4_f.get(), entry5_f.get(), entry6_f.get(), entry7_f.get()]
        fixvalue = [float(value) for value in fixvalue]

    
    
def open_parameters_window():
    global parameters_window
    global var1, var2, var3, var4, var5, var6, var7
    global entry1_l, entry1_u, entry2_l, entry2_u, entry3_l, entry3_u, entry4_l, entry4_u, entry5_l, entry5_u, entry6_l, entry6_u, entry7_l, entry7_u
    global entry1_f, entry2_f, entry3_f, entry4_f, entry5_f, entry6_f, entry7_f
    # 
    if ifloc ==1:
        boundary= bounds_l
        fixv = fixvalue_l
        print('localization')
    elif ifloc ==0:
        boundary = bounds
        fixv = fixvalue
        print('numerosity')
    else:
        print('error')

    parameters_window = tk.Toplevel(window)
    parameters_window.title("Parameters")
    parameters_window.geometry("600x300")
    
    paras_label = tk.Label(parameters_window, text="Parameters", fg="blue")
    paras_label.place(relx=0.1, rely=0.02, anchor="w")
    fix_label = tk.Label(parameters_window, text="Fixed value", fg="black")
    fix_label.place(relx=0.4, rely=0.02, anchor="center")
    low_label = tk.Label(parameters_window, text="Lower bound", fg="blue")
    low_label.place(relx=0.6, rely=0.02, anchor="center")
    up_label = tk.Label(parameters_window, text="Upper bound", fg="blue")
    up_label.place(relx=0.8, rely=0.02, anchor="center")
    
    
    # 
    var1 = tk.BooleanVar(value=True)
    var2 = tk.BooleanVar(value=True)
    var3 = tk.BooleanVar(value=True)
    var4 = tk.BooleanVar(value=True)
    var5 = tk.BooleanVar(value=True)
    var6 = tk.BooleanVar()
    var7 = tk.BooleanVar()

    check_button1 = tk.Checkbutton(parameters_window, text="pcommon", variable=var1)
    check_button2 = tk.Checkbutton(parameters_window, text="sigma1", variable=var2)
    check_button3 = tk.Checkbutton(parameters_window, text="sigma2", variable=var3)
    check_button4 = tk.Checkbutton(parameters_window, text="sigma_p", variable=var4)
    check_button5 = tk.Checkbutton(parameters_window, text="mean_p", variable=var5)
    check_button6 = tk.Checkbutton(parameters_window, text="s_1", variable=var6)
    check_button7 = tk.Checkbutton(parameters_window, text="s_2", variable=var7)

    check_button1.place(relx=0.1, rely=0.1, anchor="w")
    check_button2.place(relx=0.1, rely=0.2, anchor="w")
    check_button3.place(relx=0.1, rely=0.3, anchor="w")
    check_button4.place(relx=0.1, rely=0.4, anchor="w")
    check_button5.place(relx=0.1, rely=0.5, anchor="w")
    check_button6.place(relx=0.1, rely=0.6, anchor="w")
    check_button7.place(relx=0.1, rely=0.7, anchor="w")

    #fix
    entry1_f = tk.Entry(parameters_window, width=5)
    entry2_f = tk.Entry(parameters_window, width=5)
    entry3_f = tk.Entry(parameters_window, width=5)
    entry4_f = tk.Entry(parameters_window, width=5)
    entry5_f = tk.Entry(parameters_window, width=5)
    entry6_f = tk.Entry(parameters_window, width=5)
    entry7_f = tk.Entry(parameters_window, width=5)

    #  
    entry1_f.insert(tk.END, fixv[0])
    entry2_f.insert(tk.END, fixv[1])
    entry3_f.insert(tk.END, fixv[2])
    entry4_f.insert(tk.END, fixv[3])
    entry5_f.insert(tk.END, fixv[4])
    entry6_f.insert(tk.END, fixv[5])
    entry7_f.insert(tk.END, fixv[6])

    entry1_f.place(relx=0.4, rely=0.1, anchor="center")
    entry2_f.place(relx=0.4, rely=0.2, anchor="center")
    entry3_f.place(relx=0.4, rely=0.3, anchor="center")
    entry4_f.place(relx=0.4, rely=0.4, anchor="center")
    entry5_f.place(relx=0.4, rely=0.5, anchor="center")
    entry6_f.place(relx=0.4, rely=0.6, anchor="center")
    entry7_f.place(relx=0.4, rely=0.7, anchor="center")

    # low
    entry1_l = tk.Entry(parameters_window, width=5)
    entry2_l = tk.Entry(parameters_window, width=5)
    entry3_l = tk.Entry(parameters_window, width=5)
    entry4_l = tk.Entry(parameters_window, width=5)
    entry5_l = tk.Entry(parameters_window, width=5)
    entry6_l = tk.Entry(parameters_window, width=5)
    entry7_l = tk.Entry(parameters_window, width=5)

    #  
    entry1_l.insert(tk.END, boundary[0][0])
    entry2_l.insert(tk.END, boundary[1][0])
    entry3_l.insert(tk.END, boundary[2][0])
    entry4_l.insert(tk.END, boundary[3][0])
    entry5_l.insert(tk.END, boundary[4][0])
    entry6_l.insert(tk.END, 1)
    entry7_l.insert(tk.END, 1)

    entry1_l.place(relx=0.6, rely=0.1, anchor="center")
    entry2_l.place(relx=0.6, rely=0.2, anchor="center")
    entry3_l.place(relx=0.6, rely=0.3, anchor="center")
    entry4_l.place(relx=0.6, rely=0.4, anchor="center")
    entry5_l.place(relx=0.6, rely=0.5, anchor="center")
    entry6_l.place(relx=0.6, rely=0.6, anchor="center")
    entry7_l.place(relx=0.6, rely=0.7, anchor="center")
    

    # up
    entry1_u = tk.Entry(parameters_window, width=5)
    entry2_u = tk.Entry(parameters_window, width=5)
    entry3_u = tk.Entry(parameters_window, width=5)
    entry4_u = tk.Entry(parameters_window, width=5)
    entry5_u = tk.Entry(parameters_window, width=5)
    entry6_u = tk.Entry(parameters_window, width=5)
    entry7_u = tk.Entry(parameters_window, width=5)
    
    entry1_u.insert(tk.END, boundary[0][1])
    entry2_u.insert(tk.END, boundary[1][1])
    entry3_u.insert(tk.END, boundary[2][1])
    entry4_u.insert(tk.END, boundary[3][1])
    entry5_u.insert(tk.END, boundary[4][1])
    entry6_u.insert(tk.END, 50000)
    entry7_u.insert(tk.END, 50000)

    entry1_u.place(relx=0.8, rely=0.1, anchor="center")
    entry2_u.place(relx=0.8, rely=0.2, anchor="center")
    entry3_u.place(relx=0.8, rely=0.3, anchor="center")
    entry4_u.place(relx=0.8, rely=0.4, anchor="center")
    entry5_u.place(relx=0.8, rely=0.5, anchor="center")
    entry6_u.place(relx=0.8, rely=0.6, anchor="center")
    entry7_u.place(relx=0.8, rely=0.7, anchor="center")
    
    apply_button = tk.Button(parameters_window, text="Apply", command=save_setting)
    apply_button.place(relx=0.4, rely=0.9, anchor="center")
    done_button = tk.Button(parameters_window, text="Done", command=done_setting)
    done_button.place(relx=0.6, rely=0.9, anchor="center")
    

    
def function_a():
    print("You choose the Averaging strategy")

def function_b():
    print("You choose the Selecting strategy")

def function_c():
    print("You choose the Matching strategy")
    
def function_a_simu():
    global stra_simu
    if var_a_simu.get():
        print("You choose the Averaging strategy")
        var_b_simu.set(False) 
        var_c_simu.set(False)
        stra_simu = 1
    else:
        print("You cancel the Averaging strategy")
    
def function_b_simu():
    global stra_simu
    if var_b_simu.get():
        print("You choose the Selecting strategy")
        var_a_simu.set(False) 
        var_c_simu.set(False)
        stra_simu = 2
    else:
        print("You cancel the Selecting strategy")

def function_c_simu():
    global stra_simu
    if var_c_simu.get():
        print("You choose the Matching strategy")
        var_a_simu.set(False) 
        var_b_simu.set(False)
        stra_simu = 3
    else:
        print("You cancel the Matching strategy")
    

def plot_func(ifsave):
    # 
    selected_image = tk.IntVar()
    if ifsave==2:
        target_folder = filedialog.askdirectory()
    else:
        num_plot = tk.simpledialog.askinteger("Plot", "Please enter the position number of data you want to plot (e.g. 1 for the first one)", minvalue=0, maxvalue=10000)

    for ndata in range (len(data_matrices)):
        fig = plt.figure()
        if init_numt == True:
            [error,modelprop,dataprop,d] = simulateVV_GUI(parameters_list[ndata], 10000, data_matrices[ndata], 1, strategy = strategy_list[ndata], fitType = 'mll', es_para = checkbox_vars, fixvalue = [0.5,0.4,0.8,4000,2,70000.5,70000.5])

        else :
            [error,modelprop,dataprop,c] = simulateLCD_GUI(parameters_list[ndata], 10000, data_matrices[ndata], strategy = strategy_list[ndata], fittype = 'mll', es_para = checkbox_vars, fixvalue = [0.5,0.4,0.8,4000,2,70000.5,70000.5])

        a = dataprop.shape
        condi = int(np.ceil((a[1] + 1) ** 0.5))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  
        
        for i in range(a[1]):
            if condi < 11:
                plt.subplot(condi, condi, i + 2)
            else:
                plt.figure(int(i/condi) + 1)
                plt.subplot(condi, 1, (i % condi) + 1)
        
            plt.plot(dataprop[0, i, :], 'b', label='dataU' if i == 0 else "")
     
            plt.plot(dataprop[1, i, :], 'r', label='dataD' if i == 0 else "")
            plt.axis([0, a[2]-1, 0, 1])

        if modelprop is not None:
            for i in range(a[1]):
                if condi < 11:
                    plt.subplot(condi, condi, i+2)
                else:
                    plt.figure(int(i/condi) + 1)
                    plt.subplot(condi, 1, (i % condi) + 1)
            
           
                plt.plot(modelprop[0, i, :], 'b-.', label='modelU' if i == 0 else "")
                plt.plot(modelprop[1, i, :], 'r-.', label='modelD' if i == 0 else "")
                plt.axis([0, a[2]-1, 0, 1])
                
        fig.legend(loc='upper left', bbox_to_anchor=(0, 1))

        if ifsave == 1 and ndata == num_plot-1:
            file_path = file_paths[ndata]
            file_name = os.path.basename(file_path)
            file_name = os.path.splitext(file_name)[0]
            root = tk.Tk()
            root.title('Figure')
            label = tk.Label(root, text=f"Result for {file_name}_{strategy_list[ndata]}")
            label.pack()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
            canvas.draw()
            canvas.get_tk_widget().pack()

        elif ifsave == 2:
            if target_folder:
                file_path = file_paths[ndata]
                file_name = os.path.basename(file_path)
                file_name = os.path.splitext(file_name)[0]
                figure_name = file_name + f"_{strategy_list[ndata]}"+ ".png"
                figure_path = os.path.join(target_folder, figure_name)
                plt.savefig(figure_path)

def plot_func_l(ifsave):
    
    selected_image = tk.IntVar()
    if ifsave==2:
        target_folder = filedialog.askdirectory()
    else:
        num_plot = tk.simpledialog.askinteger("Plot", "Please enter the position number of data you want to plot (e.g. 1 for the first one)", minvalue=0, maxvalue=10000)

    for ndata in range (len(data_matrices)):
        plt.figure()
        [err, modelmat, datamat, conditions]= simulateLC_GUI(parameters_list[ndata], 10000, data_matrices[ndata], strategy = strategy_list[ndata], es_para = checkbox_vars, fixvalue = fixvalue_l)
        a = datamat.shape
        condi = int((a[1] + 1) ** 0.5)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)  
        for i in range(a[1]):
            if condi < 11:
                plt.subplot(condi, condi, i + 2)
            else:
                plt.figure(int(i/condi) + 1)
                plt.subplot(condi, 1, (i % condi) + 1)
            
            if datamat[0, i, 0] < 100:
                D1 = datamat[0, i, :]
                n, bins, _ = plt.hist(D1, bins=10, density=True, alpha=0.5, color='blue', label='Behavior Data')

            if datamat[1, i, 0] < 100:
                D2 = datamat[1, i, :]
                n, bins, _ = plt.hist(D2, bins=10, density=True, alpha=0.5, color='red', label='Behavior Data')
            
            plt.xticks([])  # Remove x-axis tick marks
            plt.yticks([])  # Remove y-axis tick marks
    
        if modelmat is not None:
            for i in range(a[1]):
                if condi < 11:
                    plt.subplot(condi, condi, i+2)
                else:
                    plt.figure(int(i/condi) + 1)
                    plt.subplot(condi, 1, (i % condi) + 1)
            
                if modelmat[0, i, 0] < 100:
                    D1 = modelmat[0, i, :]
                    kde1 = gaussian_kde(D1)
                    x1 = np.linspace(min(D1), max(D1), 100)
                    pdf1 = kde1.evaluate(x1)
                    plt.plot(x1, pdf1, color='b', linewidth=2, label='model_Distribution1')
                
                if modelmat[1, i, 0] < 100:    
            
                    D2 = modelmat[1, i, :]
                    kde2 = gaussian_kde(D2)
            
                    x2 = np.linspace(min(D2), max(D2), 100)
                    pdf2 = kde2.evaluate(x2)
                    plt.plot(x2, pdf2, color='r', linewidth=2, label='model_Distribution2')
                
                plt.axvline(x=conditions[i,0], linestyle=':', color='b')
                plt.axvline(x=conditions[i,1], linestyle=':', color='r')
            
                plt.axis([-30, 30, 0, 0.2])
                plt.xticks([])  # Remove x-axis tick marks
                plt.yticks([])  # Remove y-axis tick marks
        
        if ifsave == 1 and ndata == num_plot-1:
            file_path = file_paths[ndata]
            file_name = os.path.basename(file_path)
            file_name = os.path.splitext(file_name)[0]
            root = tk.Tk()
            root.title('Figure')
            label = tk.Label(root, text=f"Result for {file_name}_{strategy_list[ndata]}")
            label.pack()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
            canvas.draw()
            canvas.get_tk_widget().pack()

        elif ifsave == 2:
            if target_folder:
                file_path = file_paths[ndata]
                file_name = os.path.basename(file_path)
                file_name = os.path.splitext(file_name)[0]
                figure_name = file_name + f"_{strategy_list[ndata]}"+ ".png"
                figure_path = os.path.join(target_folder, figure_name)
                plt.savefig(figure_path)

                

    
def plot_Simu(ifdata):
    global canvas_sim
    plt.clf()
    # Default Parameters
    sti1 = variable_sti1.get()
    sti2 = variable_sti2.get()
    pcommon = variable_pcommon.get()
    sigmaU = variable_sigmaU.get()
    sigmaD = variable_sigmaD.get()
    sigmaZ = variable_sigmaZ.get()
    PZ_center = variable_mup.get()
    
    sU = 70000.5 
    sD = 70000.5
    sUm = 0
    sDm = 0
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    n=1000
    conditions = np.array([sti1,sti2])
    real = np.tile(conditions, (n, 1))
    
    # Create Mean of Distribution 
    # sU and sD is the 1/rate of increase away from center
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0]
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1]
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)
    
    #sigma_like[real[:, 0] == 0, 0] = 1000
    #sigma_like[real[:, 1] == 0, 1] = 1000
    #########
    # Calculate p(C|D,U)
    #########
    
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))
    #print(np.shape(PCDU))
    #########
    # Calculate Sc_hat
    #########
    
    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )
    
    # Mean Responses (Sim)
    responsesSim = np.zeros((Sc.shape[0], 2))
    if stra_simu == 1:
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
        
    elif stra_simu == 2:
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > 0.5, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > 0.5, Sc, Snc2)
    
    else:
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)

    if ifdata == 2:
        header = ["StimulusU", "StimulusD"]
        file_path_simudata = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path_simudata:
            with open(file_path_simudata, 'w') as file:
                header_line = ','.join(header) + '\n'
                file.write(header_line)
                for row in responsesSim:
                    line = ','.join(str(item) for item in row) + '\n'
                    file.write(line)
        return file_path_simudata
        
    if var_respD.get():  
        D1 = responsesSim[:, 0]
        D2 = responsesSim[:, 1]
        # kernel density estimation
        kde1 = gaussian_kde(D1)
        kde2 = gaussian_kde(D2)

        # Generate the value of the probability density function
        x1 = np.linspace(min(D1), max(D1), 100)
        pdf1 = kde1.evaluate(x1)
        x2 = np.linspace(min(D2), max(D2), 100)
        pdf2 = kde2.evaluate(x2)

        # distribution curves
        plt.plot(x1, pdf1, color='b', linewidth=2, label='Resp_Distribution1')
        plt.plot(x2, pdf2, color='r', linewidth=2, label='Resp_Distribution2')
    
    plt.axvline(x=sti1, linestyle='--', color='b')
    plt.axvline(x=sti2, linestyle='--', color='r')

    if var_stiE.get():
        Dnc1 = Snc1
        Dnc2 = Snc2
        xnc1 = np.linspace(min(Dnc1), max(Dnc1), 100)
        xnc2 = np.linspace(min(Dnc2), max(Dnc2), 100)
        # distribution curves
        plt.plot(xnc1, norm.pdf(xnc1, sti1, sigmaU), color='lightblue', linestyle='--', linewidth=2, label='Sti_Encoding1')
        plt.plot(xnc2, norm.pdf(xnc2, sti2, sigmaD), color='lightcoral', linestyle='--', linewidth=2, label='Sti_Encoding2')

    if var_priorD.get():
        D1 = responsesSim[:, 0]
        D2 = responsesSim[:, 1]
        xp = np.linspace(np.min([D1, D2]), np.max([D1, D2]), 100)
        plt.plot(xp, norm.pdf(xp, PZ_center, sigmaZ), color='lightgreen', linestyle='--', linewidth=2, label='Prior')
        
    if var_peak.get() and var_respD.get():
        peak_index_D1 = np.argmax(pdf1)
        peak_index_D2 = np.argmax(pdf2)
        peak_value_D1 = pdf1[peak_index_D1]
        peak_value_D2 = pdf2[peak_index_D2]   
        plt.scatter(x1[peak_index_D1], peak_value_D1, color='b', marker='D', s=50, label='Peak1')
        plt.scatter(x2[peak_index_D2], peak_value_D2, color='r', marker='D', s=50, label='Peak2')
        

    if var_mean.get() and var_respD.get():
        mean_D1 = np.trapz(x1 * pdf1, x1) / np.trapz(pdf1, x1)
        mean_D2 = np.trapz(x2 * pdf2, x2) / np.trapz(pdf2, x2)
        plt.scatter(mean_D1, kde1.evaluate(mean_D1), color='b', label='Mean1', marker='v', s=50)
        plt.scatter(mean_D2, kde2.evaluate(mean_D2), color='r', label='Mean2', marker='v', s=50)

    if var_disp.get() and var_peak.get() and var_respD.get():
        plt.annotate(f'{x1[peak_index_D1]:.2f}', xy=(x1[peak_index_D1], peak_value_D1), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black'))
        plt.annotate(f'{x2[peak_index_D2]:.2f}', xy=(x2[peak_index_D2], peak_value_D2), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black'))

    if var_disp.get() and var_mean.get() and var_respD.get():
        plt.annotate(f'{mean_D1:.2f}', xy=(mean_D1, kde1.evaluate(mean_D1)), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black'))
        plt.annotate(f'{mean_D2:.2f}', xy=(mean_D2, kde2.evaluate(mean_D2)), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black'))

    ax = plt.gca()


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(prop={'size': 8})
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    
    root_sim = tk.Tk()
    root_sim.title('Figure')
    label_sim = tk.Label(root_sim, text='Simulation Result')
    label_sim.pack()

    canvas_sim = FigureCanvasTkAgg(plt.gcf(), master=root_sim)
    canvas_sim.draw()
    canvas_sim.get_tk_widget().pack()

    save_button = tk.Button(root_sim, text="Save", command=save_figure)
    save_button.pack() 

def Simu_oneD(sti1, sti2, pcommon, sigma1, sigma2, sigmap, mup):
    plt.clf()
    # Default Parameters
    sti1 = sti1
    sti2 = sti2
    pcommon = pcommon
    sigmaU = sigma1
    sigmaD = sigma2
    sigmaZ = sigmap
    PZ_center = mup
    
    sU = 70000.5 
    sD = 70000.5
    sUm = 0
    sDm = 0
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    #np.random.seed(13)
    
    n=1000
    conditions = np.array([sti1,sti2])
    real = np.tile(conditions, (n, 1))
    
    # Create Mean of Distribution 
    # sU and sD is the 1/rate of increase away from center
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0]
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1]
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)
    
    #sigma_like[real[:, 0] == 0, 0] = 1000
    #sigma_like[real[:, 1] == 0, 1] = 1000
    #########
    # Calculate p(C|D,U)
    #########
    
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))
    #print(np.shape(PCDU))
    #########
    # Calculate Sc_hat
    #########
    
    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )
    
    # Mean Responses (Sim)
    responsesSim = np.zeros((Sc.shape[0], 2))
    if stra_simu == 1:
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
        
    elif stra_simu == 2:
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > 0.5, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > 0.5, Sc, Snc2)
    
    else:
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)

    
    return responsesSim, Snc1, Snc2

def plot_simu_2D():
    global canvas_sim

    response1, S11, S12 = Simu_oneD(variable_sti1.get(), variable_sti2.get(), variable_pcommon.get(), variable_sigmaU.get(), variable_sigmaD.get(), variable_sigmaZ.get(), variable_mup.get())
    response2, S21, S22 = Simu_oneD(variable_sti12.get(), variable_sti22.get(), variable_pcommon.get(), variable_sigmaU2.get(), variable_sigmaD2.get(), variable_sigmaZ2.get(), variable_mup2.get())

    a =response1[:,0]
    b =response2[:,0]
    c =response1[:,1]
    d =response2[:,1]

    lowb = np.min([a,b,c,d])
    highb = np.max([a,b,c,d])

    cdata = np.vstack((a, b))
    kde = gaussian_kde(cdata)
    x_range = np.linspace(lowb, highb, 100)
    y_range = np.linspace(lowb, highb, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    coordinates = np.vstack((x_grid.ravel(), y_grid.ravel()))

    z = kde(coordinates)
    z = z.reshape(x_grid.shape)

    peak_indices1 = np.unravel_index(np.argmax(z), z.shape)
    peak_x1 = x_range[peak_indices1[1]]
    peak_y1 = y_range[peak_indices1[0]]
    peak_value1 = z[peak_indices1]


    cdata = np.vstack((c, d))
    kde2 = gaussian_kde(cdata)

    z2 = kde2(coordinates)
    z2 = z2.reshape(x_grid.shape)

    peak_indices2 = np.unravel_index(np.argmax(z2), z2.shape)
    peak_x2 = x_range[peak_indices2[1]]
    peak_y2 = y_range[peak_indices2[0]]
    peak_value2 = z[peak_indices2]


    plt.figure(figsize=(9, 7))
    plt.scatter(variable_sti1.get(), variable_sti12.get(), color='b', label='Location1')
    plt.scatter(variable_sti2.get(), variable_sti22.get(), color='r', label='Location2')
    if var_respD.get(): 

        contour = plt.contour(x_grid, y_grid, z, levels=5, cmap='viridis')
        bar1 = plt.colorbar()
        contour2 = plt.contour(x_grid, y_grid, z2, levels=5, cmap='RdPu')
        bar2 = plt.colorbar()
        plt.clabel(contour, inline=1, fontsize=5)
        plt.clabel(contour2, inline=1, fontsize=5)

    if var_stiE.get():
        lowb = np.min([S11,S12,S21,S22])
        highb = np.max([S11,S12,S21,S22])

        '''
        cdata = np.vstack((S11, S21))
        kde_e1 = gaussian_kde(cdata)

        cdata = np.vstack((S12, S22))
        kde_e2 = gaussian_kde(cdata)

        z_e1 = kde_e1(coordinates)
        z_e1 = z_e1.reshape(x_grid.shape)

        z_e2 = kde_e2(coordinates)
        z_e2 = z_e2.reshape(x_grid.shape)
        '''
        ######
        x1 = np.random.normal(variable_sti1.get(), variable_sigmaU.get(),1000)
        y1 = np.random.normal(variable_sti12.get(), variable_sigmaU2.get(),1000)

        x2 = np.random.normal(variable_sti2.get(), variable_sigmaD.get(),1000)
        y2 = np.random.normal(variable_sti22.get(), variable_sigmaD2.get(),1000)

        cdata = np.vstack((x1, y1))
        kde_e1 = gaussian_kde(cdata)

        cdata = np.vstack((x2, y2))
        kde_e2 = gaussian_kde(cdata)

        z_e1 = kde_e1(coordinates)
        z_e1 = z_e1.reshape(x_grid.shape)

        z_e2 = kde_e2(coordinates)
        z_e2 = z_e2.reshape(x_grid.shape)


        contour_e1 = plt.contour(x_grid, y_grid, z_e1, levels=5, cmap='Blues', alpha=0.6, linestyles='dashed')
        contour_e2 = plt.contour(x_grid, y_grid, z_e2, levels=5, cmap='Reds', alpha=0.6, linestyles='dashed')



    if var_priorD.get():
        samples1 = np.random.normal(variable_mup.get(), variable_sigmaZ.get(), 100)
        samples2 = np.random.normal(variable_mup2.get(), variable_sigmaZ2.get(), 100)

        cdata = np.vstack((samples1, samples2))
        kde_p = gaussian_kde(cdata)

        z_p = kde_p(coordinates)
        z_p = z_p.reshape(x_grid.shape)

        contour_p = plt.contour(x_grid, y_grid, z_p, levels=5, cmap='YlOrBr', alpha=0.6, linestyles='dashed')
        
    if var_peak.get() and var_respD.get():
        plt.scatter(peak_x1, peak_y1, color='b',marker='D', label='Peak1')
        plt.scatter(peak_x2, peak_y2, color='r',marker='D', label='Peak2')
        
    if var_mean.get() and var_respD.get():
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        mean_c = np.mean(c)
        mean_d = np.mean(d)
        
        plt.scatter(mean_a, mean_b, color='b', label='Mean1', marker='v', s=50)
        plt.scatter(mean_c, mean_d, color='r', label='Mean2', marker='v', s=50)

    if var_disp.get() and var_peak.get() and var_respD.get():
        plt.annotate(f'({peak_x1:.2f}, {peak_y1:.2f})', xy=(peak_x1, peak_y1), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black', alpha=0.1))
        plt.annotate(f'({peak_x2:.2f}, {peak_y2:.2f})', xy=(peak_x2, peak_y2), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black', alpha=0.1))
       
    if var_disp.get() and var_mean.get() and var_respD.get():
        plt.annotate(f'({mean_a:.2f}, {mean_b:.2f})', xy=(mean_a, mean_b), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black', alpha=0.1))
        plt.annotate(f'({mean_c:.2f}, {mean_d:.2f})', xy=(mean_c, mean_d), xytext=None,
             arrowprops=dict(arrowstyle='-', color='black', alpha=0.1))
             


    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel(entry_dimen1.get())
    plt.ylabel(entry_dimen2.get())
    plt.title('Simulation for 2-D Continuous Condition')
    plt.legend(prop={'size': 8})
    root_sim = tk.Tk()
    
    
    root_sim.title('Figure')
    label_sim = tk.Label(root_sim, text='Simulation Result')
    label_sim.pack()

    canvas_sim = FigureCanvasTkAgg(plt.gcf(), master=root_sim)
    canvas_sim.draw()
    canvas_sim.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    save_button = tk.Button(root_sim, text="Save", command=save_figure)
    save_button.pack()   



def save_2D():
    response1,k,kk = Simu_oneD(variable_sti1.get(), variable_sti2.get(), variable_pcommon.get(), variable_sigmaU.get(), variable_sigmaD.get(), variable_sigmaZ.get(), variable_mup.get())
    response2,k,kk = Simu_oneD(variable_sti12.get(), variable_sti22.get(), variable_pcommon.get(), variable_sigmaU2.get(), variable_sigmaD2.get(), variable_sigmaZ2.get(), variable_mup2.get())

    a =response1[:,0]
    b =response2[:,0]
    c =response1[:,1]
    d =response2[:,1]
    responsesSim = [a,b,c,d]
    responsesSim=np.transpose(responsesSim)
    header = ["Spatial_1", "Temporal_1", "Spatial_2", "Temporal_2"]
    file_path_simudata = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path_simudata:
        with open(file_path_simudata, 'w') as file:
            header_line = ','.join(header) + '\n'
            file.write(header_line)
            for row in responsesSim:
                line = ','.join(str(item) for item in row) + '\n'
                file.write(line)
    

def plot_Simu_disc(ifdata):
    global canvas_sim
    plt.clf()
    # Default Parameters
    sti1 = variable_sti1.get()
    sti2 = variable_sti2.get()
    pcommon = variable_pcommon.get()
    sigmaU = variable_sigmaU.get()
    sigmaD = variable_sigmaD.get()
    sigmaZ = variable_sigmaZ.get()
    PZ_center = variable_mup.get()
    
    sU = 70000.5 
    sD = 70000.5
    sUm = 0
    sDm = 0
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    n=1000
    conditions = np.array([sti1,sti2])
    real = np.tile(conditions, (n, 1))
    
    # Create Mean of Distribution 
    # sU and sD is the 1/rate of increase away from center
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0]
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1]
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)
    
    #sigma_like[real[:, 0] == 0, 0] = 1000
    #sigma_like[real[:, 1] == 0, 1] = 1000
    #sigma_like[real[:, 0] == 0, 1] = 0.001
    #sigma_like[real[:, 1] == 0, 0] = 0.001
    #########
    # Calculate p(C|D,U)
    #########
    
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))
    #print(np.shape(PCDU))
    #########
    # Calculate Sc_hat
    #########
    
    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )
    
    # Mean Responses (Sim)
    responsesSim = np.zeros((Sc.shape[0], 2))
    if stra_simu == 1:
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
        
    elif stra_simu == 2:
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > 0.5, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > 0.5, Sc, Snc2)
    
    else:
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)

    if ifdata == 2:
        header = ["StimulusU", "StimulusD"]
        file_path_simudata = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path_simudata:
            with open(file_path_simudata, 'w') as file:
                header_line = ','.join(header) + '\n'
                file.write(header_line)
                for row in np.maximum(np.round(responsesSim),0):
                    line = ','.join(str(item) for item in row) + '\n'
                    file.write(line)
        return file_path_simudata
        
    if var_respD.get():  
        D1 = np.maximum(np.round(responsesSim[:, 0]),0)
        D2 = np.maximum(np.round(responsesSim[:, 1]),0)
        
        values_D1, counts_D1 = np.unique(D1, return_counts=True)
        values_D2, counts_D2 = np.unique(D2, return_counts=True)

        prob_density_D1 = counts_D1 / (len(D1) * 1)
        prob_density_D2 = counts_D2 / (len(D2) * 1)

        # Generate the value of the probability density function
        bars1 = plt.bar(values_D1, prob_density_D1, align='center', color='blue', alpha=0.7, width=0.7, label='Resp_Distribution1 1')
        bars2 = plt.bar(values_D2, prob_density_D2, align='center', color='red', alpha=0.5, width=0.5, label='Resp_Distribution2 2')
        
        min_val = min(min(D1), min(D2))
        max_val = max(max(D1), max(D2))
        plt.xticks(np.arange(min_val, max_val + 1, step=1))


    
    plt.axvline(x=sti1, linestyle='--', color='b', label='Stimulus 1', linewidth=2)
    plt.axvline(x=sti2, linestyle='--', color='r', label='Stimulus 2', linewidth=2)

    if var_stiE.get():
        Dnc1 = Snc1
        Dnc2 = Snc2
        xnc1 = np.linspace(min(Dnc1), max(Dnc1), 100)
        xnc2 = np.linspace(min(Dnc2), max(Dnc2), 100)
        # distribution curves
        plt.plot(xnc1, norm.pdf(xnc1, sti1, sigmaU), color='lightblue', linestyle='--', linewidth=2, label='Sti_Encoding1')
        plt.plot(xnc2, norm.pdf(xnc2, sti2, sigmaD), color='lightcoral', linestyle='--', linewidth=2, label='Sti_Encoding2')

    if var_priorD.get():
        D1 = responsesSim[:, 0]
        D2 = responsesSim[:, 1]
        xp = np.linspace(np.min([D1, D2]), np.max([D1, D2]), 100)
        plt.plot(xp, norm.pdf(xp, PZ_center, sigmaZ), color='lightgreen', linestyle='--', linewidth=2, label='Prior')
        
    if var_peak.get() and var_respD.get():
        peak_index_D1 = np.argmax(prob_density_D1)
        peak_index_D2 = np.argmax(prob_density_D2)

        peak_value_D1 = values_D1[peak_index_D1]
        peak_value_D2 = values_D2[peak_index_D2]


        plt.scatter(peak_value_D1, prob_density_D1[peak_index_D1], marker='D', color='blue', s=50, label='Peak1')
        plt.scatter(peak_value_D2, prob_density_D2[peak_index_D2], marker='D', color='red', s=50, label='Peak2')

        

    if var_disp.get() and var_peak.get() and var_respD.get():
        #plt.annotate(f'{prob_density_D1[peak_index_D1]:.2f}',  
             #(peak_value_D1, prob_density_D1[peak_index_D1]), 
             #textcoords="offset points", 
             #xytext=(0,10),  
             #ha='center') 


        #plt.annotate(f'{prob_density_D2[peak_index_D2]:.2f}', 
             #(peak_value_D2, prob_density_D2[peak_index_D2]),  
             #textcoords="offset points",  
             #xytext=(0,10), 
             #ha='center')  


        for bar, pd in zip(bars1, prob_density_D1):
            plt.annotate(f'{pd:.2f}',
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 textcoords="offset points",
                 xytext=(0,5),  
                 ha='center', va='bottom')

        for bar, pd in zip(bars2, prob_density_D2):
            plt.annotate(f'{pd:.2f}',
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 textcoords="offset points",
                 xytext=(0,5),  
                 ha='center', va='bottom')

    ax = plt.gca()


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(3) 
    ax.spines['left'].set_linewidth(3) 
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)  

    plt.legend(prop={'size': 8})
    plt.xlabel('Value', fontsize=13, fontweight='bold')
    plt.ylabel('Probability', fontsize=13, fontweight='bold')
    
    root_sim = tk.Tk()
    root_sim.title('Figure')
    label_sim = tk.Label(root_sim, text='Simulation Result')
    label_sim.pack()

    canvas_sim = FigureCanvasTkAgg(plt.gcf(), master=root_sim)
    canvas_sim.draw()
    canvas_sim.get_tk_widget().pack()

    save_button = tk.Button(root_sim, text="Save", command=save_figure)
    save_button.pack()  


def save_figure():
    # 弹出文件保存对话框
    filetypes = [
        ("PNG files", "*.png"),
        ("PDF files", "*.pdf"),
        ("SVG files", "*.svg"),
        ("EPS files", "*.eps"),
        ("TIFF files", "*.tif"),
        ("All files", "*.*")
    ]
    filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=filetypes)
    if not filepath:
        return
    # 保存图形
    canvas_sim.figure.savefig(filepath, dpi=300)  # 指定分辨率为300DPI


#Generate fake data
def generate_combinations(n):
    combinations = [[i, j] for i in range(n) for j in range(n)]
    combinations = np.array(combinations[1:], dtype=float)
    return combinations

def save_p():
    file_list = []
    header_list = ['pcommon', 'sigmaU', 'sigmaD', 'sigmap', 'mup', 'sU', 'sD']
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        file_list.append(file_name)

    result_list = list(zip(file_list, parameters_list, error_list, strategy_list))
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        header = "File Name\t" + "\t".join(np.array(header_list)[checkbox_vars ==1 ]) + "\tError\tStrategy\n"
        with open(file_path, "w") as txtfile:
            txtfile.write(header)
            for rdata in result_list:
                arr_str = '\t'.join(f"{element:.6f}" for element in rdata[1])
                txtfile.write(f"{rdata[0]}\t{arr_str}\t{rdata[2]}\t{rdata[3]}\n")

def plot_Simuall():
    Nums = variable_Num.get()
    pcommon = variable_pcommon.get()
    sigmaU = variable_sigmaU.get()
    sigmaD = variable_sigmaD.get()
    sigmaZ = variable_sigmaZ.get()
    PZ_center = variable_mup.get()
    
    

    fdata = generate_combinations(int(Nums))
    repeated_combinations = np.repeat(fdata, 2, axis=0)
    fdata = np.reshape(repeated_combinations, (len(fdata), 4))
    if stra_simu==1:
        stra = 'ave'
    elif stra_simu ==2:
        stra = 'sel'
    else:
        stra = 'mat'
    
    [error, modelprop, dataprop,resi] = simulateVV([pcommon,sigmaU,sigmaD,sigmaZ,PZ_center], 10000, fdata, biOnly = 1,  strategy = stra , fitType = 'dif')
   
    a = dataprop.shape
    condi = int((a[1] + 1) ** 0.5)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  


    if modelprop is not None:
        for i in range(a[1]):
            if condi < 11:
                plt.subplot(condi, condi, i+2)
            else:
                plt.figure(int(i/condi) + 1)
                plt.subplot(condi, 1, (i % condi) + 1)
            
           
            plt.plot(modelprop[0, i, :], 'b-.')
            plt.plot(modelprop[1, i, :], 'r-.')
            plt.axis([0, a[2]-1, 0, 1])
    
    
    root = tk.Tk()
    root.title('Figure')
    label = tk.Label(root, text='Simulation Result')
    label.pack()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    

def run_code():
    global result_label
    global progressbar
    global start_time
    global parameters_list, error_list, strategy_list

    nseeds = int(entry_seeds.get())
    
    strategies = []
    if var_a.get():
        strategies.append(('ave', 'Averaging'))
    if var_b.get():
        strategies.append(('sel', 'Selection'))
    if var_c.get():
        strategies.append(('mat', 'Matching'))

    if len(strategies) == 0:
        result_label.config(text="Please select at least one strategy.", fg="red")
        return

    if var_mll.get():
        fit_t = 'mll'
    elif var_mr2.get():
        fit_t = 'mr2'
    else:
        fit_t = 'sse'

    parameters_list = []
    error_list = []
    strategy_list = []
    finalR = ""
    

    for ndata in range(len(data_matrices)):
        # Create progress bar (need to check)
        progressbar = ttk.Progressbar(window, mode="determinate", maximum=100)
        progressbar.pack()
        data = data_matrices[ndata]

        best_error = float('inf')
        global best_result
        global plot_strategy
        best_result = None
        best_strategy = None
        plot_strategy = None

        for randseed in range(randseeds):
            np.random.seed(randseed_mat[randseed])
            x0 = np.random.rand(count)

        
            for strategy, strategy_name in strategies:
                # Show "Please wait..."
                result_label.config(text=f"Fitting with {strategy_name} strategy & {fit_t} fit type for data {ndata+1}... (randseed = {randseed_mat[randseed]})", fg="blue")
                window.update()
                if init_P ==1:

                    def minimize_callback(xk):
                        estim_t = (nseeds * 0.0035)*count/5
                        cur_t = time.time()
                        # Compute progress
                        progress = (cur_t - start_time )/ estim_t
                        print(progress)
                        # Refresh bar
                        progressbar["value"] = progress * 100
            
                        window.update()
        
                    start_time = time.time()
                    if init_numt == True:
                        result = minimize(lambda paras: simulateVV_GUI(paras, nseeds, data, 0, strategy=strategy, fitType=fit_t, es_para = checkbox_vars, fixvalue = fixvalue)[0],
                          x0, method='Powell', bounds=bounds_use, callback=minimize_callback)
                    else:
                        result = minimize(lambda paras: simulateLCD_GUI(paras, nseeds, data, strategy=strategy, fittype=fit_t, es_para = checkbox_vars, fixvalue = fixvalue)[0],
                          x0, method='Powell', bounds=bounds_use, callback=minimize_callback)


                    if result.success and result.fun < best_error:
                        best_error = result.fun
                        best_result = result.x
                        best_strategy = strategy_name
                        plot_strategy = strategy

                if init_V ==1:
                    def minimize_callback(xk):
                        estim_t = (nseeds * 0.35)*count/5
                        cur_t = time.time()
                        # Compute progress
                        progress = (cur_t - start_time )/ estim_t
                        print(progress)
                        # Refresh bar
                        progressbar["value"] = progress * 100
            
                        window.update()

                    start_time = time.time()
                    x0 = x0.reshape(1, 5)
                    LB = np.array([tup[0]-0.01 for tup in bounds_use])
                    LB = LB.reshape(1, 5)
                    UB = np.array([tup[1]+0.01 for tup in bounds_use]) # Upper bounds
                    UB = UB.reshape(1, 5)
                    PLB = np.array([tup[0] for tup in bounds_use])
                    PLB = PLB.reshape(1, 5)
                    PUB = np.array([tup[1] for tup in bounds_use]) 
                    PUB = PUB.reshape(1, 5)
                    x1 = x0 * [PUB-PLB] + PLB
                    x1 = x1.reshape(-1)

                    if init_numt == True:
                        vbmc = VBMC(lambda paras: -simulateVV_GUI(paras, nseeds, data, 0, strategy=strategy, fitType=fit_t, es_para = checkbox_vars, fixvalue = fixvalue)[0], x1, LB, UB, PLB, PUB)
                    else:
                        vbmc = VBMC(lambda paras: -simulateLCD_GUI(paras, nseeds, data, strategy=strategy, fittype=fit_t, es_para = checkbox_vars, fixvalue = fixvalue)[0], x1, LB, UB, PLB, PUB)
                    
                    #vbmc.register_callback(minimize_callback)
                    vp, results = vbmc.optimize()
                    n_samples = int(3e5)
                    Xs, _ = vp.sample(n_samples)
                    # Easily compute statistics such as moments, credible intervals, etc.
                    post_mean = np.mean(Xs, axis=0)  # Posterior mean
                    if results['elbo'] < best_error:
                        best_error = results['elbo']
                        best_result = post_mean
                        best_strategy = strategy_name
                        plot_strategy = strategy

       

        if best_result is not None:
            # Get result2.x & result2.fun
            result_x = str(best_result)
            result_fun = str(best_error)

            # Show "Result is ... (fun)"
            text = f"Best result for data {ndata+1} is {result_x}\nError is {result_fun}\nStrategy: {best_strategy}"
            finalR = finalR + '\n' + text
        else:
            result_label.config(text="No valid results found.", fg="red")

        # Hide bar
        progressbar.pack_forget()
        parameters_list.append(best_result)
        error_list.append(result_fun)
        strategy_list.append(plot_strategy)

    result_window = tk.Tk()
    result_window.title('Estimated Parameters')

    def on_scroll(*args):
        text_box.yview(*args)

    scrollbar = tk.Scrollbar(result_window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_box = tk.Text(result_window, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text_box.pack(expand=True, fill=tk.BOTH)
    scrollbar.config(command=text_box.yview)
    text_box.insert(tk.END, finalR)

    result_label.config(text= 'Finished' , fg="black")
    save_paras = tk.Button(result_window, text="Save", command=save_p)
    save_paras.pack()

def run_code_l():
    global result_label
    global progressbar
    global start_time
    global parameters_list, error_list, strategy_list

    nseeds = int(entry_seeds.get())
    
    strategies = []
    if var_a.get():
        strategies.append(('ave', 'Averaging'))
    if var_b.get():
        strategies.append(('sel', 'Selection'))
    if var_c.get():
        strategies.append(('mat', 'Matching'))

    if len(strategies) == 0:
        result_label.config(text="Please select at least one strategy.", fg="red")
        return

    if var_mll.get():
        fit_t = 'fre'
    elif var_mr2.get():
        fit_t = 'mr2'
    else:
        fit_t = 'emd'


    parameters_list = []
    error_list = []
    strategy_list = []
    finalR = ""
    for ndata in range(len(data_matrices)):
        # Create progress bar (need to check)
        progressbar = ttk.Progressbar(window, mode="determinate", maximum=100)
        progressbar.pack()
        data = data_matrices[ndata]

        best_error = float('inf')
        global best_result
        global plot_strategy
        best_result = None
        best_strategy = None
        plot_strategy = None

        for randseed in range(randseeds):
            np.random.seed(randseed_mat[randseed])
            x0 = np.random.rand(count)

            for strategy, strategy_name in strategies:
                # Show "Please wait..."
                result_label.config(text=f"Fitting with {strategy_name} strategy & {fit_t} fit type for data {ndata+1}...", fg="blue")
                window.update()
                if init_P ==1:
        
                    def minimize_callback(xk):
                        estim_t = (nseeds * 0.100)*count/5
                        cur_t = time.time()
                        # Compute progress
                        progress = (cur_t - start_time )/ estim_t
                        print(progress)
                        # Refresh bar
                        progressbar["value"] = progress * 100
                        window.update()
        
        
                    start_time = time.time()

                    result = minimize(lambda paras: simulateLC_GUI(paras, nseeds, data, strategy=strategy,fittype=fit_t, es_para = checkbox_vars, fixvalue = fixvalue_l)[0],
                          x0, method='Powell', bounds=bounds_l_use, callback=minimize_callback)


                    if result.success and result.fun < best_error:
                        best_error = result.fun
                        best_result = result.x
                        best_strategy = strategy_name
                        plot_strategy = strategy

                if init_V ==1:
                    def minimize_callback(xk):
                        estim_t = (nseeds * 0.35)*count/5
                        cur_t = time.time()
                        # Compute progress
                        progress = (cur_t - start_time )/ estim_t
                        print(progress)
                        # Refresh bar
                        progressbar["value"] = progress * 100
            
                        window.update()

                    start_time = time.time()
                    x0 = x0.reshape(1, 5)
                    LB = np.array([tup[0]-0.01 for tup in bounds_l_use])
                    LB = LB.reshape(1, 5)
                    UB = np.array([tup[1]+0.01 for tup in bounds_l_use]) # Upper bounds
                    UB = UB.reshape(1, 5)
                    PLB = np.array([tup[0] for tup in bounds_l_use])
                    PLB = PLB.reshape(1, 5)
                    PUB = np.array([tup[1] for tup in bounds_l_use]) 
                    PUB = PUB.reshape(1, 5)
                    x1= x0 * [PUB-PLB] + PLB
                    x1 = x1.reshape(-1)
                    print(x1)
                    vbmc = VBMC(lambda paras: -simulateLC_GUI(paras, nseeds, data, strategy=strategy,fittype=fit_t, es_para = checkbox_vars, fixvalue = fixvalue_l)[0], x1, LB, UB, PLB, PUB)
                    #vbmc.register_callback(minimize_callback)
                    vp, results = vbmc.optimize()
                    n_samples = int(3e5)
                    Xs, _ = vp.sample(n_samples)
                    # Easily compute statistics such as moments, credible intervals, etc.
                    post_mean = np.mean(Xs, axis=0)  # Posterior mean
                    if results['elbo'] < best_error:
                        best_error = results['elbo']
                        best_result = post_mean
                        best_strategy = strategy_name
                        plot_strategy = strategy


        if best_result is not None:
            # Get result2.x & result2.fun
            result_x = str(best_result)
            result_fun = str(best_error)

            # Show "Result is ... (fun)"
            text = f"Best result for data {ndata+1} is {result_x}\nError is {result_fun}\nStrategy: {best_strategy}"
            finalR = finalR + '\n' + text
        else:
            result_label.config(text="No valid results found.", fg="red")

        # Hide bar
        progressbar.pack_forget()
        parameters_list.append(best_result)
        error_list.append(result_fun)
        strategy_list.append(plot_strategy)

    result_window = tk.Tk()
    result_window.title('Estimated Parameters')

    def on_scroll(*args):
        text_box.yview(*args)

    scrollbar = tk.Scrollbar(result_window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_box = tk.Text(result_window, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text_box.pack(expand=True, fill=tk.BOTH)
    scrollbar.config(command=text_box.yview)
    text_box.insert(tk.END, finalR)

    result_label.config(text= 'Finished' , fg="black")
    save_paras = tk.Button(result_window, text="Save", command=save_p)
    save_paras.pack()

    
    
def fit_page():
    global var_a, ifloc, bounds_use
    global var_b
    global var_c
    global var_mll
    global var_mr2
    global var_sse
    global result_label
    global entry_seeds, entry
    global fit_frame

    ifloc = 0
    
    bounds_use = bounds[:count]
    # Create homepage framework
    fit_frame = tk.Frame(window)
    welcome_frame.pack_forget()  # Hide Welcome
    modelfit_frame.pack_forget()
    fit_frame.pack(fill="both", expand=True)  # Show Homepage
    # Create Text Box
    entry = tk.Entry(fit_frame, width=60)
    entry.pack()

    button = tk.Button(fit_frame, text="Import", command=lambda: import_csv(entry))
    button.place(relx=0.4, rely=0.07, anchor="center")
    button2 = tk.Button(fit_frame, text="Open file", command=lambda: open_file(entry))
    button2.place(relx=0.6, rely=0.07, anchor="center")
    demo_button = tk.Button(fit_frame, text="demo", command=demo)
    demo_button.place(relx=0.9, rely=0.07, anchor="center")

    help_button = tk.Button(fit_frame, text="help", command=show_help2)
    help_button.place(relx=0.9, rely=0.11, anchor="center")
    
    seeds_label = tk.Label(fit_frame, text="Number of simulations : ", fg="black")
    seeds_label.place(relx=0.4, rely=0.12, anchor="center")
    
    entry_seeds = tk.Entry(fit_frame, width = 5)
    entry_seeds.insert(tk.END, 1000)
    entry_seeds.place(relx=0.6, rely=0.12, anchor="center")

    # Create checkboxes and corresponding functions
    var_a = tk.BooleanVar()
    var_b = tk.BooleanVar()
    var_c = tk.BooleanVar()

    check_button_a = tk.Checkbutton(fit_frame, text="Model Averaging", variable=var_a, command=function_a)
    check_button_b = tk.Checkbutton(fit_frame, text="Model Selection", variable=var_b, command=function_b)
    check_button_c = tk.Checkbutton(fit_frame, text="Probability Matching", variable=var_c, command=function_c)

    strategy_label = tk.Label(fit_frame, text="select at least one strategy", fg="dark green")
    strategy_label.place(relx=0.7, rely=0.15, anchor="center")
    check_button_a.place(relx=0.7, rely=0.2, anchor="center")
    check_button_b.place(relx=0.695, rely=0.25, anchor="center")
    check_button_c.place(relx=0.72, rely=0.3, anchor="center")

    # Create checkboxes and corresponding functions
    var_mll = tk.BooleanVar(value=True)
    var_mr2 = tk.BooleanVar()
    var_sse = tk.BooleanVar()

    check_button_mll = tk.Checkbutton(fit_frame, text="mll", variable=var_mll)
    check_button_mr2 = tk.Checkbutton(fit_frame, text="mr2", variable=var_mr2)
    check_button_sse = tk.Checkbutton(fit_frame, text="sse", variable=var_sse)

    type_label = tk.Label(fit_frame, text="select one fit type", fg="dark green")
    type_label.place(relx=0.3, rely=0.15, anchor="center")
    check_button_mll.place(relx=0.3, rely=0.2, anchor="center")
    check_button_mr2.place(relx=0.3, rely=0.25, anchor="center")
    check_button_sse.place(relx=0.3, rely=0.3, anchor="center")

    
    parameters_button = tk.Button(fit_frame, text="Parameters", command=open_parameters_window)
    parameters_button.place(relx=0.5, rely=0.4, anchor="center")
    
    run_button = tk.Button(fit_frame, text="Run", command=run_code)
    run_button.place(relx=0.5, rely=0.45, anchor="center")
    # Show results' labels
    result_label = tk.Label(fit_frame, text= "Result will be here...: ", fg="gray")
    result_label.place(relx=0.5, rely=0.5, anchor="center")
    
    # Plot
    plot_button = tk.Button(fit_frame, text="Plot", command=lambda: plot_func(1))
    plot_button.place(relx=0.5, rely=0.55, anchor="center")

    # Save
    save_fit = tk.Button(fit_frame, text="Save", command=lambda: plot_func(2))
    save_fit.place(relx=0.5, rely=0.6, anchor="center")

    
    # Back
    Back_but = tk.Button(fit_frame, text="Main Page", fg="green", command=back_main1)
    Back_but.place(relx=0.5, rely=0.7, anchor="center")

def fit_page_l():
    global var_a, ifloc, bounds_l_use
    global var_b
    global var_c
    global var_mll
    global var_mr2
    global var_emd
    global result_label
    global entry_seeds
    global fit_frame

    ifloc = 1
    bounds_l_use = bounds_l[:count]
    # Create homepage framework
    fit_frame = tk.Frame(window)
    welcome_frame.pack_forget() 
    modelfit_frame.pack_forget() # Hide Welcome
    fit_frame.pack(fill="both", expand=True)  # Show Homepage
    # Create Text Box
    entry = tk.Entry(fit_frame, width=60)
    entry.pack()

    button = tk.Button(fit_frame, text="Import", command=lambda: import_csv2(entry))
    button.place(relx=0.4, rely=0.07, anchor="center")
    button2 = tk.Button(fit_frame, text="Open file", command=lambda: open_file2(entry))
    button2.place(relx=0.6, rely=0.07, anchor="center")
    
    seeds_label = tk.Label(fit_frame, text="Number of simulations : ", fg="black")
    seeds_label.place(relx=0.4, rely=0.12, anchor="center")
    
    entry_seeds = tk.Entry(fit_frame, width = 5)
    entry_seeds.insert(tk.END, 1000)
    entry_seeds.place(relx=0.6, rely=0.12, anchor="center")

    # Create checkboxes and corresponding functions
    var_a = tk.BooleanVar()
    var_b = tk.BooleanVar()
    var_c = tk.BooleanVar()

    check_button_a = tk.Checkbutton(fit_frame, text="Model Averaging", variable=var_a, command=function_a)
    check_button_b = tk.Checkbutton(fit_frame, text="Model Selection", variable=var_b, command=function_b)
    check_button_c = tk.Checkbutton(fit_frame, text="Probability Matching", variable=var_c, command=function_c)

    strategy_label = tk.Label(fit_frame, text="select at least one strategy", fg="dark green")
    strategy_label.place(relx=0.7, rely=0.15, anchor="center")
    check_button_a.place(relx=0.7, rely=0.2, anchor="center")
    check_button_b.place(relx=0.695, rely=0.25, anchor="center")
    check_button_c.place(relx=0.72, rely=0.3, anchor="center")

    # Create checkboxes and corresponding functions
    var_mll = tk.BooleanVar(value=True)
    var_emd = tk.BooleanVar()
    var_mr2 = tk.BooleanVar()
    
    check_button_mll = tk.Checkbutton(fit_frame, text="mll", variable=var_mll)
    check_button_mr2 = tk.Checkbutton(fit_frame, text="emd", variable=var_emd)
    check_button_sse = tk.Checkbutton(fit_frame, text="mr2", variable=var_mr2)

    type_label = tk.Label(fit_frame, text="select one fit type", fg="dark green")
    type_label.place(relx=0.3, rely=0.15, anchor="center")
    check_button_mll.place(relx=0.3, rely=0.2, anchor="center")
    check_button_mr2.place(relx=0.3, rely=0.25, anchor="center")
    check_button_sse.place(relx=0.3, rely=0.3, anchor="center")
 
    parameters_button = tk.Button(fit_frame, text="Parameters", command=open_parameters_window)
    parameters_button.place(relx=0.5, rely=0.4, anchor="center")
    
    run_button = tk.Button(fit_frame, text="Run", command=run_code_l)
    run_button.place(relx=0.5, rely=0.45, anchor="center")
    # Show results' labels
    result_label = tk.Label(fit_frame, text= "Result will be here...: ", fg="gray")
    result_label.place(relx=0.5, rely=0.5, anchor="center")
    
    # Plot
    plot_button = tk.Button(fit_frame, text="Plot", command=lambda: plot_func_l(1))
    plot_button.place(relx=0.5, rely=0.55, anchor="center")

    # Save
    save_fit = tk.Button(fit_frame, text="Save", command=lambda: plot_func_l(2))
    save_fit.place(relx=0.5, rely=0.6, anchor="center")
    help_button = tk.Button(fit_frame, text="help", command=show_help2)
    help_button.place(relx=0.9, rely=0.11, anchor="center")
    
    # Back
    Back_but = tk.Button(fit_frame, text="Main Page", fg="green", command=back_main1)
    Back_but.place(relx=0.5, rely=0.7, anchor="center")


    
def back_main():
    simu_frame.pack_forget()
    open_welcome()
    
def back_main1():
    fit_frame.pack_forget()
    open_welcome()

def back_main2():
    modelfit_frame.pack_forget()
    open_welcome()

def back_main3():
    simu_frame.pack_forget()
    open_welcome()


    

def update_value(value):
      
    try:
        entry_pcommon.insert(variable_pcommon.get())
        entry_sigmaU.insert(variable_sigmaU)
        entry_sigmaD.insert(variable_sigmaD)
        entry_sigmaZ.insert(variable_sigmaZ)
        entry_sti1.insert(variable_sti1)
        entry_sti2.insert(variable_sti2)
        
    except Exception as e:
        pass  
    

def entry_changed(event):
    # 输入框的回调函数，用于更新滑轨的值
    value_pcommon = entry_pcommon.get()
    scale_pcommon.set(value_pcommon)
    
    value_sigmaU = entry_sigmaU.get()
    scale_sigmaU.set(value_sigmaU)
    
    value_sigmaD = entry_sigmaD.get()
    scale_sigmaD.set(value_sigmaD)
    
    value_sigmaZ = entry_sigmaZ.get()
    scale_sigmaZ.set(value_sigmaZ)
    
    value_sti1 = entry_sti1.get()
    scale_sti1.set(value_sti1)
    
    value_sti2 = entry_sti2.get()
    scale_sti2.set(value_sti2)
    
def simu_page(singleif):
    global simu_frame, variable_pcommon, variable_sigmaU, variable_sigmaD, variable_sigmaZ, variable_mup, variable_sti1, variable_sti2, variable_Num
    global entry_pcommon, entry_sigmaU, entry_sigmaD, entry_sigmaZ, entry_mup, entry_sti1, entry_sti2
    global scale_pcommon, scale_sigmaU, scale_sigmaD, scale_sigmaZ, scale_mup, scale_sti1, scale_sti2
    global var_a_simu, var_b_simu, var_c_simu, var_respD, var_stiE, var_priorD, var_peak, var_mean, var_disp
    global stra_simu
    welcome_frame.pack_forget()  # Hide Welcome
    simu_frame.pack_forget()
    simu_frame = tk.Frame(window)
    simu_frame.pack()  # Show Homepage
    stra_simu = 1
    variable_pcommon = tk.DoubleVar()
    variable_pcommon.set(0.5)
    variable_sigmaU = tk.DoubleVar()
    variable_sigmaU.set(1)
    variable_sigmaD = tk.DoubleVar()
    variable_sigmaD.set(0.5)
    variable_sigmaZ = tk.DoubleVar()
    variable_sigmaZ.set(1.5)
    variable_mup = tk.DoubleVar()
    variable_mup.set(2)
    variable_sti1 = tk.DoubleVar()
    variable_sti1.set(1)
    variable_sti2 = tk.DoubleVar()
    variable_sti2.set(2)
    variable_Num = tk.DoubleVar()
    variable_Num.set(3)
    
    
    

    
    scale_pcommon = tk.Scale(simu_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_pcommon, command=update_value, length=500)
    scale_pcommon.pack()
    
    entry_pcommon = tk.Entry(simu_frame, textvariable=variable_pcommon, width=5)
    entry_pcommon.pack()
    entry_pcommon.bind('<Return>', entry_changed)
    
    scale_sigmaU = tk.Scale(simu_frame, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaU, command=update_value, length=500)
    scale_sigmaU.pack()
    
    entry_sigmaU = tk.Entry(simu_frame, textvariable=variable_sigmaU, width=5)
    entry_sigmaU.pack()
    entry_sigmaU.bind('<Return>', entry_changed)
    
    scale_sigmaD = tk.Scale(simu_frame, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaD, command=update_value, length=500)
    scale_sigmaD.pack()
    
    entry_sigmaD = tk.Entry(simu_frame, textvariable=variable_sigmaD, width=5)
    entry_sigmaD.pack()
    entry_sigmaD.bind('<Return>', entry_changed)
    
    scale_sigmaZ = tk.Scale(simu_frame, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaZ, command=update_value, length=500)
    scale_sigmaZ.pack()
    
    entry_sigmaZ = tk.Entry(simu_frame, textvariable=variable_sigmaZ, width=5)
    entry_sigmaZ.pack()
    entry_sigmaZ.bind('<Return>', entry_changed)
    
    scale_mup = tk.Scale(simu_frame, from_=-30, to=30, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_mup, command=update_value, length=500)
    scale_mup.pack()
    
    entry_mup = tk.Entry(simu_frame, textvariable=variable_mup, width=5)
    entry_mup.pack()
    entry_mup.bind('<Return>', entry_changed)
    if singleif==1:
        scale_sti1 = tk.Scale(simu_frame, from_=-40, to=40, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=variable_sti1, command=update_value, length=500)
        scale_sti1.pack()
    
        entry_sti1 = tk.Entry(simu_frame, textvariable=variable_sti1, width=5)
        entry_sti1.pack()
        entry_sti1.bind('<Return>', entry_changed)
    
        scale_sti2 = tk.Scale(simu_frame, from_=-40, to=40, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=variable_sti2, command=update_value, length=500)
        scale_sti2.pack()
    
        entry_sti2 = tk.Entry(simu_frame, textvariable=variable_sti2, width=5)
        entry_sti2.pack()
        entry_sti2.bind('<Return>', entry_changed)
    
        
        pcommon_label = tk.Label(simu_frame, text="Pcommon", fg="black")
        pcommon_label.place(relx=0.3, rely=0.08, anchor="w")
    
        sigmaU_label = tk.Label(simu_frame, text="\u03c3\u2081", font=("Arial", 18), fg="blue")
        sigmaU_label.place(relx=0.37, rely=0.19, anchor="w")
    
        sigmaD_label = tk.Label(simu_frame, text="\u03c3\u2082", font=("Arial", 18), fg="red")
        sigmaD_label.place(relx=0.37, rely=0.29, anchor="w")
    
        sigmaZ_label = tk.Label(simu_frame, text="\u03c3p", font=("Arial", 18), fg="black")
        sigmaZ_label.place(relx=0.37, rely=0.395, anchor="w")
    
        mup_label = tk.Label(simu_frame, text="\u03bcp", font=("Arial", 18), fg="black")
        mup_label.place(relx=0.37, rely=0.5, anchor="w")
    
    
        sti1_label = tk.Label(simu_frame, text="stimulus 1", fg="blue")
        sti1_label.place(relx=0.3, rely=0.6, anchor="w")
    
        sti2_label = tk.Label(simu_frame, text="stimulus 2", fg="red")
        sti2_label.place(relx=0.3, rely=0.7, anchor="w")
        
        placeholder = tk.Frame(simu_frame, height=90)
        placeholder.pack()
        # Plot
        plot_Simu_but = tk.Button(simu_frame, text="Simulate", command= lambda: plot_Simu(1))
        plot_Simu_but.pack()

        save_Simu_but = tk.Button(simu_frame, text="Save Simulated Data", command= lambda: plot_Simu(2))
        save_Simu_but.pack()

        var_respD = tk.BooleanVar(value=True)
        var_stiE = tk.BooleanVar()
        var_priorD = tk.BooleanVar()

        button_respD = tk.Checkbutton(simu_frame, text="Response Distribution", variable=var_respD)
        button_stiE= tk.Checkbutton(simu_frame, text="Stimulus Encoding", variable=var_stiE)
        button_prior = tk.Checkbutton(simu_frame, text="Prior Distribution", variable=var_priorD)
    
        button_respD.place(relx=0.35, rely=0.75, anchor="e")
        button_stiE.place(relx=0.65, rely=0.75, anchor="e")
        button_prior.place(relx=0.95, rely=0.75, anchor="e")  

        var_peak = tk.BooleanVar(value=True)
        var_mean = tk.BooleanVar()
        var_disp = tk.BooleanVar()

        button_peak = tk.Checkbutton(simu_frame, text="Peak", variable=var_peak)
        button_mean = tk.Checkbutton(simu_frame, text="Mean", variable=var_mean)
        button_disp = tk.Checkbutton(simu_frame, text="Display Value", variable=var_disp)
    
        button_peak.place(relx=0.15, rely=0.85, anchor="e")
        button_mean.place(relx=0.5, rely=0.85, anchor="e")
        button_disp.place(relx=0.91, rely=0.85, anchor="e")  
        help_button = tk.Button(simu_frame, text="Help", command=show_help1)
        help_button.place(relx=0.2, rely=0.92, anchor="e")    
        
        
    else:
        
        entry_N = tk.Entry(simu_frame, textvariable=variable_Num, width=3)
        entry_N.pack()
    
        # Plot
        placeholder = tk.Frame(simu_frame, height=50)
        placeholder.pack()
        plot_Simuall_but = tk.Button(simu_frame, text="Simulate", command=plot_Simuall)
        plot_Simuall_but.pack()
        pcommon_label = tk.Label(simu_frame, text="Pcommon", fg="black")
        pcommon_label.place(relx=0.37, rely=0.11, anchor="w")
    
        sigmaU_label = tk.Label(simu_frame, text="\u03c3\u2081", font=("Arial", 18), fg="blue")
        sigmaU_label.place(relx=0.37, rely=0.26, anchor="w")
    
        sigmaD_label = tk.Label(simu_frame, text="\u03c3\u2082", font=("Arial", 18), fg="red")
        sigmaD_label.place(relx=0.37, rely=0.4, anchor="w")
    
        sigmaZ_label = tk.Label(simu_frame, text="\u03c3p", font=("Arial", 18), fg="black")
        sigmaZ_label.place(relx=0.37, rely=0.54, anchor="w")
    
        mup_label = tk.Label(simu_frame, text="\u03bcp", font=("Arial", 18), fg="black")
        mup_label.place(relx=0.37, rely=0.68, anchor="w")
        Num_label = tk.Label(simu_frame, text="Number of stimuli", fg="black")
        Num_label.place(relx=0.3, rely=0.75, anchor="center")
                
    var_a_simu = tk.BooleanVar(value=True)
    var_b_simu = tk.BooleanVar()
    var_c_simu = tk.BooleanVar()

    check_button_a_simu = tk.Checkbutton(simu_frame, text="Model Averaging", variable=var_a_simu, command=function_a_simu)
    check_button_b_simu = tk.Checkbutton(simu_frame, text="Model Selection", variable=var_b_simu, command=function_b_simu)
    check_button_c_simu = tk.Checkbutton(simu_frame, text="Probability Matching", variable=var_c_simu, command=function_c_simu)
    
    check_button_a_simu.place(relx=0.3, rely=0.8, anchor="e")
    check_button_b_simu.place(relx=0.62, rely=0.8, anchor="e")
    check_button_c_simu.place(relx=0.99, rely=0.8, anchor="e") 

     
    # Back
    plot_Simu_but = tk.Button(simu_frame, text="Main Page", fg="green", command=back_main)
    plot_Simu_but.pack()

def open_link():
    
    url = "https://bci-toolbox.readthedocs.io/en/latest/index.html" 

    webbrowser.open(url)

def show_help():
    
    help_window = tk.Toplevel(simu_frame)
    help_window.title("Help")

    
   

    original_image = Image.open("resources/inst2D.png")  
    #resized_image = original_image.resize((1200, 700), Image.ANTIALIAS)
    original_image.thumbnail((1600, 900))

    photo = ImageTk.PhotoImage(original_image)
    
    image_label = tk.Label(help_window, image=photo)
    image_label.image = photo  
    image_label.pack()
    custom_font = ("Arial", 24, "bold")

    help_label = tk.Label(help_window, text="For more information and details, please click here to check the documentation.", fg="black", cursor="hand2", font=custom_font)
    help_label.pack()

    help_label.bind("<Button-1>", lambda e: open_link())

def show_help1():
    
    help_window = tk.Toplevel(simu_frame)
    help_window.title("Help")

    original_image = Image.open("resources/inst1D.png")  
    #resized_image = original_image.resize((1200, 700), Image.ANTIALIAS)
    original_image.thumbnail((1600, 900))

    photo = ImageTk.PhotoImage(original_image)
    
    image_label = tk.Label(help_window, image=photo)
    image_label.image = photo  
    image_label.pack()
    custom_font = ("Arial", 24, "bold")

    help_label = tk.Label(help_window, text="For more information and details, please click here to check the documentation.", fg="black", cursor="hand2", font=custom_font)
    help_label.pack()

    help_label.bind("<Button-1>", lambda e: open_link())

def show_help2():
    
    help_window = tk.Toplevel(fit_frame)
    help_window.title("Help")

    
   

    original_image = Image.open("resources/instdisc.png")  
    #resized_image = original_image.resize((1000, 700), Image.ANTIALIAS)
    original_image.thumbnail((1400, 800))

    photo = ImageTk.PhotoImage(original_image)
    
    image_label = tk.Label(help_window, image=photo)
    image_label.image = photo  
    image_label.pack()
    custom_font = ("Arial", 24, "bold")

    help_label = tk.Label(help_window, text="For more information and details, please click here to check the documentation.", fg="black", cursor="hand2", font=custom_font)
    help_label.pack()

    help_label.bind("<Button-1>", lambda e: open_link())

def simupage_2D():
    global simu_frame, variable_pcommon, variable_sigmaU, variable_sigmaD, variable_sigmaZ, variable_mup, variable_sti1, variable_sti2, variable_Num
    global variable_sigmaU2, variable_sigmaD2, variable_sigmaZ2, variable_mup2, variable_sti12, variable_sti22
    global entry_pcommon, entry_sigmaU, entry_sigmaD, entry_sigmaZ, entry_mup, entry_sti1, entry_sti2
    global scale_pcommon, scale_sigmaU, scale_sigmaD, scale_sigmaZ, scale_mup, scale_sti1, scale_sti2
    global var_a_simu, var_b_simu, var_c_simu, var_respD, var_stiE, var_priorD, var_peak, var_mean, var_disp
    global stra_simu
    global entry_dimen1, entry_dimen2
    welcome_frame.pack_forget()  # Hide Welcome
    simu_frame.pack_forget()
    simu_frame = tk.Frame(window)
    simu_frame.pack(fill="both", expand=True) 
    stra_simu = 1
    variable_pcommon = tk.DoubleVar()
    variable_pcommon.set(0.5)
    variable_sigmaU = tk.DoubleVar()
    variable_sigmaU.set(10)
    variable_sigmaD = tk.DoubleVar()
    variable_sigmaD.set(5)
    variable_sigmaZ = tk.DoubleVar()
    variable_sigmaZ.set(15)
    variable_mup = tk.DoubleVar()
    variable_mup.set(0)
    variable_sti1 = tk.DoubleVar()
    variable_sti1.set(-10)
    variable_sti2 = tk.DoubleVar()
    variable_sti2.set(10)

    variable_sigmaU2 = tk.DoubleVar()
    variable_sigmaU2.set(10)
    variable_sigmaD2 = tk.DoubleVar()
    variable_sigmaD2.set(5)
    variable_sigmaZ2 = tk.DoubleVar()
    variable_sigmaZ2.set(15)
    variable_mup2 = tk.DoubleVar()
    variable_mup2.set(0)
    variable_sti12 = tk.DoubleVar()
    variable_sti12.set(-10)
    variable_sti22 = tk.DoubleVar()
    variable_sti22.set(10)

    canvas = tk.Canvas(simu_frame, width=300, height=300)
    canvas.place(relx=0.3, rely=0.3, anchor="e")

    
    

    # Set Location
    scale_pcommon = tk.Scale(simu_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_pcommon, command=update_value, length=500)
    scale_pcommon.pack()
    
    entry_pcommon = tk.Entry(simu_frame, textvariable=variable_pcommon, width=5)
    entry_pcommon.pack()
    entry_pcommon.bind('<Return>', entry_changed)

    left_column = tk.Frame(simu_frame)
    left_column.pack(side=tk.LEFT)


    scale_sigmaU = tk.Scale(left_column, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaU, command=update_value, length=200)
    scale_sigmaU.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sigmaU = tk.Entry(left_column, textvariable=variable_sigmaU, width=5)
    entry_sigmaU.pack(side=tk.TOP, padx=10, pady=3)
    entry_sigmaU.bind('<Return>', entry_changed)
    
    scale_sigmaD = tk.Scale(left_column, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaD, command=update_value, length=200)
    scale_sigmaD.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sigmaD = tk.Entry(left_column, textvariable=variable_sigmaD, width=5)
    entry_sigmaD.pack(side=tk.TOP, padx=10, pady=3)
    entry_sigmaD.bind('<Return>', entry_changed)
    
    scale_sigmaZ = tk.Scale(left_column, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaZ, command=update_value, length=200)
    scale_sigmaZ.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sigmaZ = tk.Entry(left_column, textvariable=variable_sigmaZ, width=5)
    entry_sigmaZ.pack(side=tk.TOP, padx=10, pady=3)
    entry_sigmaZ.bind('<Return>', entry_changed)
    
    scale_mup = tk.Scale(left_column, from_=-30, to=30, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_mup, command=update_value, length=200)
    scale_mup.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_mup = tk.Entry(left_column, textvariable=variable_mup, width=5)
    entry_mup.pack(side=tk.TOP, padx=10, pady=3)
    entry_mup.bind('<Return>', entry_changed)
    
    
    scale_sti1 = tk.Scale(left_column, from_=-40, to=40, resolution=0.1, orient=tk.HORIZONTAL,
    variable=variable_sti1, command=update_value, length=200)
    scale_sti1.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sti1 = tk.Entry(left_column, textvariable=variable_sti1, width=5)
    entry_sti1.pack(side=tk.TOP, padx=10, pady=3)
    entry_sti1.bind('<Return>', entry_changed)
    
    scale_sti2 = tk.Scale(left_column, from_=-40, to=40, resolution=0.1, orient=tk.HORIZONTAL,
    variable=variable_sti2, command=update_value, length=200)
    scale_sti2.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sti2 = tk.Entry(left_column, textvariable=variable_sti2, width=5)
    entry_sti2.pack(side=tk.TOP, padx=10, pady=3)
    entry_sti2.bind('<Return>', entry_changed)
    
    georgia_font = ("Georgia", 18, "normal")   
    pcommon_label = tk.Label(simu_frame, text="Pcommon", fg="black")
    pcommon_label.place(relx=0.33, rely=0.07, anchor="w")

    entry_dimen1 = tk.Entry(simu_frame, width=7, fg="purple")
    entry_dimen1.insert(0, "Spatial")
    entry_dimen1.place(relx=0.18, rely=0.08, anchor="w")

    spatial_label = tk.Label(simu_frame, text="Dimension1:", fg="black", font=("Arial", 12))
    spatial_label.place(relx=0.05, rely=0.08, anchor="w")

    entry_dimen2 = tk.Entry(simu_frame, width=7, fg="purple")
    entry_dimen2.insert(0, "Temporal")
    entry_dimen2.place(relx=0.8, rely=0.08, anchor="w")

    temporal_label = tk.Label(simu_frame, text="Dimension2:", fg="black", font=("Arial", 12))
    temporal_label.place(relx=0.68, rely=0.08, anchor="w")
    
    sigmaU_label = tk.Label(left_column, text="\u03c3\u2081", font=("Arial", 18), fg="blue")
    sigmaU_label.place(relx=0.2, rely=0.095, anchor="w")
    
    sigmaD_label = tk.Label(left_column, text="\u03c3\u2082", font=("Arial", 18), fg="red")
    sigmaD_label.place(relx=0.2, rely=0.22, anchor="w")
    
    sigmaZ_label = tk.Label(left_column, text="\u03c3p", font=("Arial", 18), fg="black")
    sigmaZ_label.place(relx=0.2, rely=0.345, anchor="w")
    
    mup_label = tk.Label(left_column, text="\u03bcp", font=("Arial", 18), fg="black")
    mup_label.place(relx=0.2, rely=0.47, anchor="w")
    
    sti1_label = tk.Label(left_column, text="stimulus 1", fg="blue")
    sti1_label.place(relx=0.08, rely=0.595, anchor="w")
    
    sti2_label = tk.Label(left_column, text="stimulus 2", fg="red")
    sti2_label.place(relx=0.08, rely=0.72, anchor="w")
        
    placeholder = tk.Frame(left_column, height=220)
    placeholder.pack()

    # Right
    right_column = tk.Frame(simu_frame)
    right_column.pack(side=tk.RIGHT)

    scale_sigmaU2 = tk.Scale(right_column, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaU2, command=update_value, length=200)
    scale_sigmaU2.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sigmaU2 = tk.Entry(right_column, textvariable=variable_sigmaU2, width=5)
    entry_sigmaU2.pack(side=tk.TOP, padx=10, pady=3)
    entry_sigmaU2.bind('<Return>', entry_changed)
    
    scale_sigmaD2 = tk.Scale(right_column, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaD2, command=update_value, length=200)
    scale_sigmaD2.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sigmaD2 = tk.Entry(right_column, textvariable=variable_sigmaD2, width=5)
    entry_sigmaD2.pack(side=tk.TOP, padx=10, pady=3)
    entry_sigmaD2.bind('<Return>', entry_changed)
    
    scale_sigmaZ2 = tk.Scale(right_column, from_=0.1, to=35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaZ2, command=update_value, length=200)
    scale_sigmaZ2.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sigmaZ2 = tk.Entry(right_column, textvariable=variable_sigmaZ2, width=5)
    entry_sigmaZ2.pack(side=tk.TOP, padx=10, pady=3)
    entry_sigmaZ2.bind('<Return>', entry_changed)
    
    scale_mup2 = tk.Scale(right_column, from_=-30, to=30, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_mup2, command=update_value, length=200)
    scale_mup2.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_mup2 = tk.Entry(right_column, textvariable=variable_mup2, width=5)
    entry_mup2.pack(side=tk.TOP, padx=10, pady=3)
    entry_mup2.bind('<Return>', entry_changed)
    
    
    scale_sti12 = tk.Scale(right_column, from_=-40, to=40, resolution=0.1, orient=tk.HORIZONTAL,
    variable=variable_sti12, command=update_value, length=200)
    scale_sti12.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sti12 = tk.Entry(right_column, textvariable=variable_sti12, width=5)
    entry_sti12.pack(side=tk.TOP, padx=10, pady=3)
    entry_sti12.bind('<Return>', entry_changed)
    
    scale_sti22 = tk.Scale(right_column, from_=-40, to=40, resolution=0.1, orient=tk.HORIZONTAL,
    variable=variable_sti22, command=update_value, length=200)
    scale_sti22.pack(side=tk.TOP, padx=10, pady=3)
    
    entry_sti22 = tk.Entry(right_column, textvariable=variable_sti22, width=5)
    entry_sti22.pack(side=tk.TOP, padx=10, pady=3)
    entry_sti22.bind('<Return>', entry_changed)
    
    
    sigmaU_label = tk.Label(right_column, text="\u03c3\u2081", font=("Arial", 18), fg="blue")
    sigmaU_label.place(relx=0.2, rely=0.095, anchor="w")
    
    sigmaD_label = tk.Label(right_column, text="\u03c3\u2082", font=("Arial", 18), fg="red")
    sigmaD_label.place(relx=0.2, rely=0.22, anchor="w")
    
    sigmaZ_label = tk.Label(right_column, text="\u03c3p", font=("Arial", 18), fg="black")
    sigmaZ_label.place(relx=0.2, rely=0.345, anchor="w")
    
    mup_label = tk.Label(right_column, text="\u03bcp", font=("Arial", 18), fg="black")
    mup_label.place(relx=0.2, rely=0.47, anchor="w")
    
    sti1_label = tk.Label(right_column, text="stimulus 1", fg="blue")
    sti1_label.place(relx=0.08, rely=0.595, anchor="w")
    
    sti2_label = tk.Label(right_column, text="stimulus 2", fg="red")
    sti2_label.place(relx=0.08, rely=0.72, anchor="w")

    placeholder = tk.Frame(right_column, height=220)
    placeholder.pack()
        

    placeholder = tk.Frame(simu_frame, height=550)
    placeholder.pack()
    # Plot
    plot_Simu_but = tk.Button(simu_frame, text="Simulate", command= plot_simu_2D)
    plot_Simu_but.pack()

    save_Simu_but = tk.Button(simu_frame, text="Save Simulated Data", command= save_2D)
    save_Simu_but.pack()

    var_respD = tk.BooleanVar(value=True)
    var_stiE = tk.BooleanVar()
    var_priorD = tk.BooleanVar()

    button_respD = tk.Checkbutton(simu_frame, text="Response Distribution", variable=var_respD)
    button_stiE= tk.Checkbutton(simu_frame, text="Stimulus Encoding", variable=var_stiE)
    button_prior = tk.Checkbutton(simu_frame, text="Prior Distribution", variable=var_priorD)
    
    button_respD.place(relx=0.33, rely=0.78, anchor="e")
    button_stiE.place(relx=0.63, rely=0.78, anchor="e")
    button_prior.place(relx=0.93, rely=0.78, anchor="e")  

    var_peak = tk.BooleanVar(value=True)
    var_mean = tk.BooleanVar()
    var_disp = tk.BooleanVar()

    button_peak = tk.Checkbutton(simu_frame, text="Peak", variable=var_peak)
    button_mean = tk.Checkbutton(simu_frame, text="Mean", variable=var_mean)
    button_disp = tk.Checkbutton(simu_frame, text="Display Value", variable=var_disp)
    
    button_peak.place(relx=0.15, rely=0.86, anchor="e")
    button_mean.place(relx=0.5, rely=0.86, anchor="e")
    button_disp.place(relx=0.9, rely=0.86, anchor="e")  

    var_a_simu = tk.BooleanVar(value=True)
    var_b_simu = tk.BooleanVar()
    var_c_simu = tk.BooleanVar()

    check_button_a_simu = tk.Checkbutton(simu_frame, text="Model Averaging", variable=var_a_simu, command=function_a_simu)
    check_button_b_simu = tk.Checkbutton(simu_frame, text="Model Selection", variable=var_b_simu, command=function_b_simu)
    check_button_c_simu = tk.Checkbutton(simu_frame, text="Probability Matching", variable=var_c_simu, command=function_c_simu)
    
    check_button_a_simu.place(relx=0.27, rely=0.82, anchor="e")
    check_button_b_simu.place(relx=0.6, rely=0.82, anchor="e")
    check_button_c_simu.place(relx=0.97, rely=0.82, anchor="e") 

    help_button = tk.Button(simu_frame, text="Help", command=show_help)
    help_button.place(relx=0.2, rely=0.92, anchor="e")
    # Back
    plot_back_but = tk.Button(simu_frame, text="Main Page", fg="green", command=back_main)
    plot_back_but.pack()

    
        
    '''   
    else:
        
        entry_N = tk.Entry(simu_frame, textvariable=variable_Num, width=3)
        entry_N.pack()
    
        # Plot
        placeholder = tk.Frame(simu_frame, height=50)
        placeholder.pack()
        plot_Simuall_but = tk.Button(simu_frame, text="Simulate", command=plot_Simuall)
        plot_Simuall_but.pack()
        pcommon_label = tk.Label(simu_frame, text="Pcommon", fg="black")
        pcommon_label.place(relx=0.3, rely=0.11, anchor="w")
    
        sigmaU_label = tk.Label(simu_frame, text="sigmaU", fg="blue")
        sigmaU_label.place(relx=0.3, rely=0.26, anchor="w")
    
        sigmaD_label = tk.Label(simu_frame, text="sigmaD", fg="red")
        sigmaD_label.place(relx=0.3, rely=0.4, anchor="w")
    
        sigmaZ_label = tk.Label(simu_frame, text="sigmaZ", fg="black")
        sigmaZ_label.place(relx=0.3, rely=0.54, anchor="w")
    
        mup_label = tk.Label(simu_frame, text="Z_center", fg="black")
        mup_label.place(relx=0.3, rely=0.68, anchor="w")
        Num_label = tk.Label(simu_frame, text="Number of stimuli", fg="black")
        Num_label.place(relx=0.3, rely=0.75, anchor="center")
                
    

'''
def function_P():
    global init_V,init_P
    if var_Powell.get():
        print("You choose to use Powell algorithm to fit the model.")
        
        var_VBMC.set(False)
        init_P = True
        init_V = False

def simu_page_disc():
    global simu_frame, variable_pcommon, variable_sigmaU, variable_sigmaD, variable_sigmaZ, variable_mup, variable_sti1, variable_sti2, variable_Num
    global entry_pcommon, entry_sigmaU, entry_sigmaD, entry_sigmaZ, entry_mup, entry_sti1, entry_sti2
    global scale_pcommon, scale_sigmaU, scale_sigmaD, scale_sigmaZ, scale_mup, scale_sti1, scale_sti2
    global var_a_simu, var_b_simu, var_c_simu, var_respD, var_stiE, var_priorD, var_peak, var_mean, var_disp
    global stra_simu
    welcome_frame.pack_forget()  # Hide Welcome
    simu_frame.pack_forget()
    simu_frame = tk.Frame(window)
    simu_frame.pack()  # Show Homepage
    stra_simu = 1
    variable_pcommon = tk.DoubleVar()
    variable_pcommon.set(0.5)
    variable_sigmaU = tk.DoubleVar()
    variable_sigmaU.set(1)
    variable_sigmaD = tk.DoubleVar()
    variable_sigmaD.set(0.5)
    variable_sigmaZ = tk.DoubleVar()
    variable_sigmaZ.set(1.5)
    variable_mup = tk.DoubleVar()
    variable_mup.set(2)
    variable_sti1 = tk.DoubleVar()
    variable_sti1.set(1)
    variable_sti2 = tk.DoubleVar()
    variable_sti2.set(2)
    variable_Num = tk.DoubleVar()
    variable_Num.set(3)
    
    
    

    
    scale_pcommon = tk.Scale(simu_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_pcommon, command=update_value, length=500)
    scale_pcommon.pack()
    
    entry_pcommon = tk.Entry(simu_frame, textvariable=variable_pcommon, width=5)
    entry_pcommon.pack()
    entry_pcommon.bind('<Return>', entry_changed)
    
    scale_sigmaU = tk.Scale(simu_frame, from_=0.1, to=15, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaU, command=update_value, length=500)
    scale_sigmaU.pack()
    
    entry_sigmaU = tk.Entry(simu_frame, textvariable=variable_sigmaU, width=5)
    entry_sigmaU.pack()
    entry_sigmaU.bind('<Return>', entry_changed)
    
    scale_sigmaD = tk.Scale(simu_frame, from_=0.1, to=15, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaD, command=update_value, length=500)
    scale_sigmaD.pack()
    
    entry_sigmaD = tk.Entry(simu_frame, textvariable=variable_sigmaD, width=5)
    entry_sigmaD.pack()
    entry_sigmaD.bind('<Return>', entry_changed)
    
    scale_sigmaZ = tk.Scale(simu_frame, from_=0.1, to=15, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaZ, command=update_value, length=500)
    scale_sigmaZ.pack()
    
    entry_sigmaZ = tk.Entry(simu_frame, textvariable=variable_sigmaZ, width=5)
    entry_sigmaZ.pack()
    entry_sigmaZ.bind('<Return>', entry_changed)
    
    scale_mup = tk.Scale(simu_frame, from_=0, to=10, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_mup, command=update_value, length=500)
    scale_mup.pack()
    
    entry_mup = tk.Entry(simu_frame, textvariable=variable_mup, width=5)
    entry_mup.pack()
    entry_mup.bind('<Return>', entry_changed)
    if True:
        scale_sti1 = tk.Scale(simu_frame, from_=0, to=8, resolution=1, orient=tk.HORIZONTAL,
                 variable=variable_sti1, command=update_value, length=500)
        scale_sti1.pack()
    
        entry_sti1 = tk.Entry(simu_frame, textvariable=variable_sti1, width=5)
        entry_sti1.pack()
        entry_sti1.bind('<Return>', entry_changed)
    
        scale_sti2 = tk.Scale(simu_frame, from_=0, to=8, resolution=1, orient=tk.HORIZONTAL,
                 variable=variable_sti2, command=update_value, length=500)
        scale_sti2.pack()
    
        entry_sti2 = tk.Entry(simu_frame, textvariable=variable_sti2, width=5)
        entry_sti2.pack()
        entry_sti2.bind('<Return>', entry_changed)
    
        
        pcommon_label = tk.Label(simu_frame, text="Pcommon", fg="black")
        pcommon_label.place(relx=0.3, rely=0.08, anchor="w")
    
        sigmaU_label = tk.Label(simu_frame, text="\u03c3\u2081", font=("Arial", 18), fg="blue")
        sigmaU_label.place(relx=0.37, rely=0.19, anchor="w")
    
        sigmaD_label = tk.Label(simu_frame, text="\u03c3\u2082", font=("Arial", 18), fg="red")
        sigmaD_label.place(relx=0.37, rely=0.29, anchor="w")
    
        sigmaZ_label = tk.Label(simu_frame, text="\u03c3p", font=("Arial", 18), fg="black")
        sigmaZ_label.place(relx=0.37, rely=0.395, anchor="w")
    
        mup_label = tk.Label(simu_frame, text="\u03bcp", font=("Arial", 18), fg="black")
        mup_label.place(relx=0.37, rely=0.5, anchor="w")
    
    
        sti1_label = tk.Label(simu_frame, text="stimulus 1", fg="blue")
        sti1_label.place(relx=0.3, rely=0.6, anchor="w")
    
        sti2_label = tk.Label(simu_frame, text="stimulus 2", fg="red")
        sti2_label.place(relx=0.3, rely=0.7, anchor="w")
        
        placeholder = tk.Frame(simu_frame, height=90)
        placeholder.pack()
        # Plot
        plot_Simu_but = tk.Button(simu_frame, text="Simulate", command= lambda: plot_Simu_disc(1))
        plot_Simu_but.pack()

        save_Simu_but = tk.Button(simu_frame, text="Save Simulated Data", command= lambda: plot_Simu_disc(2))
        save_Simu_but.pack()

        var_respD = tk.BooleanVar(value=True)
        var_stiE = tk.BooleanVar()
        var_priorD = tk.BooleanVar()

        button_respD = tk.Checkbutton(simu_frame, text="Response Distribution", variable=var_respD)
        button_stiE= tk.Checkbutton(simu_frame, text="Stimulus Encoding", variable=var_stiE)
        button_prior = tk.Checkbutton(simu_frame, text="Prior Distribution", variable=var_priorD)
    
        button_respD.place(relx=0.35, rely=0.75, anchor="e")
        button_stiE.place(relx=0.65, rely=0.75, anchor="e")
        button_prior.place(relx=0.95, rely=0.75, anchor="e")  

        var_peak = tk.BooleanVar(value=True)
        var_mean = tk.BooleanVar()
        var_disp = tk.BooleanVar()

        button_peak = tk.Checkbutton(simu_frame, text="Peak", variable=var_peak)
        
        button_disp = tk.Checkbutton(simu_frame, text="Display Value", variable=var_disp)
    
        button_peak.place(relx=0.15, rely=0.85, anchor="e")
       
        button_disp.place(relx=0.91, rely=0.85, anchor="e")  
        help_button = tk.Button(simu_frame, text="Help", command=show_help1)
        help_button.place(relx=0.2, rely=0.92, anchor="e")    
        
        
    
                
    var_a_simu = tk.BooleanVar(value=True)
    var_b_simu = tk.BooleanVar()
    var_c_simu = tk.BooleanVar()

    check_button_a_simu = tk.Checkbutton(simu_frame, text="Model Averaging", variable=var_a_simu, command=function_a_simu)
    check_button_b_simu = tk.Checkbutton(simu_frame, text="Model Selection", variable=var_b_simu, command=function_b_simu)
    check_button_c_simu = tk.Checkbutton(simu_frame, text="Probability Matching", variable=var_c_simu, command=function_c_simu)
    
    check_button_a_simu.place(relx=0.3, rely=0.8, anchor="e")
    check_button_b_simu.place(relx=0.62, rely=0.8, anchor="e")
    check_button_c_simu.place(relx=0.99, rely=0.8, anchor="e") 

     
    # Back
    plot_Simu_but = tk.Button(simu_frame, text="Main Page", fg="green", command=back_main)
    plot_Simu_but.pack()


def function_V():
    global init_V,init_P
    if var_VBMC.get():
        print("You choose to use VBMC to fit the model. (Note that it requires extremely high computing resources!")
        var_Powell.set(False) 
        
        init_P = False
        init_V = True

def function_nor():
    global init_nor,init_numt
    if var_nor.get():
        print("You choose to use Normal algorithm for discrete data fitting.")
        
        var_numt.set(False)
        init_nor = True
        init_numt = False

def function_numt():
    global init_nor,init_numt
    if var_numt.get():
        print("You choose to use optimization algorithm for numerosity task. ")
        var_nor.set(False) 
        
        init_nor = False
        init_numt = True

def set_com():
    global entry_ran, set_frame
    global var_Powell, var_VBMC
    global var_numt, var_nor

    set_frame = tk.Toplevel(window)
    set_frame.title("Settings")
    set_frame.geometry("600x300")
    entry_ran = tk.Entry(set_frame, width=5)
    entry_ran.insert(tk.END, randseeds)
    entry_ran.place(relx=0.4, rely=0.1, anchor="center")

    var_Powell = tk.BooleanVar(value=init_P)
    var_VBMC = tk.BooleanVar(value=init_V)

    var_nor = tk.BooleanVar(value=init_nor)
    var_numt = tk.BooleanVar(value=init_numt)
    
    check_button_var_Powell = tk.Checkbutton(set_frame, text="Powell", variable=var_Powell, command=function_P)
    check_button_var_VBMC = tk.Checkbutton(set_frame, text="VBMC", variable=var_VBMC, command=function_V)
    
    check_button_var_Powell.place(relx=0.4, rely=0.25, anchor="center")
    check_button_var_VBMC.place(relx=0.6, rely=0.25, anchor="center")
    
    randseed_label = tk.Label(set_frame, text="Number of seeds:")
    randseed_label.place(relx=0.23, rely=0.1, anchor="center")

    fittingstra_label = tk.Label(set_frame, text="Fitting method:")
    fittingstra_label.place(relx=0.23, rely=0.25, anchor="center")

    check_button_nor = tk.Checkbutton(set_frame, text="Normal", variable=var_nor, command=function_nor)
    check_button_numt = tk.Checkbutton(set_frame, text="Numerosity Task", variable=var_numt, command=function_numt)

    check_button_nor.place(relx=0.45, rely=0.4, anchor="center")
    check_button_numt.place(relx=0.65, rely=0.4, anchor="center")

    spe_label = tk.Label(set_frame, text="Dicrete data fitting setting:")
    spe_label.place(relx=0.23, rely=0.4, anchor="center")
    

    Done_button = tk.Button(set_frame, text="Done", command=Done_set)
    Done_button.place(relx=0.5, rely=0.9, anchor="center")

def Done_set():
    global randseeds
    randseeds = entry_ran.get()
    randseeds = int(randseeds)
    set_frame.destroy()

def open_modelfit():
    global modelfit_frame
    modelfit_frame = tk.Frame(window)
    welcome_frame.pack_forget()  # Hide Welcome
    modelfit_frame.pack(fill="both", expand=True)  # Show Homepage

    startFit_button = tk.Button(modelfit_frame, text="Fitting for Discrete Data", command=fit_page, width=30, height=4)
    startFit_button.place(relx=0.5, rely=0.2, anchor="center")

    Fit2_button = tk.Button(modelfit_frame, text="Fitting for Continuous Data", command=fit_page_l,width=30, height=4)
    Fit2_button.place(relx=0.5, rely=0.35, anchor="center")

    
    setting_button = tk.Button(modelfit_frame, text="Setting", command=set_com, width=20, height=2)
    setting_button.place(relx=0.5, rely=0.55, anchor="center")

    # Back
    Back_but = tk.Button(modelfit_frame, text="Main Page", fg="green", command=back_main2)
    Back_but.place(relx=0.5, rely=0.8, anchor="center")

def open_simu():
    global simu_frame
    simu_frame = tk.Frame(window)
    welcome_frame.pack_forget()  # Hide Welcome
    simu_frame.pack(fill="both", expand=True)  # Show Homepage

    startSim_button = tk.Button(simu_frame, text="Simulating for 1-D Continuous Condition", command=lambda: simu_page(1), width=30, height=2)
    startSim_button.place(relx=0.5, rely=0.3, anchor="center")

    startSim2D_button = tk.Button(simu_frame, text="Simulating for 2-D Continuous Condition", command=simupage_2D, width=30, height=2)
    startSim2D_button.place(relx=0.5, rely=0.4, anchor="center")


    startSimall_button = tk.Button(simu_frame, text="Simulating for Numerosity Task", command=lambda: simu_page(2), width=30, height=2)
    startSimall_button.place(relx=0.5, rely=0.5, anchor="center")

    startSimdis_button = tk.Button(simu_frame, text="Simulating for Discrete Condition", command=simu_page_disc, width=30, height=2)
    startSimdis_button.place(relx=0.5, rely=0.6, anchor="center")

    # Back
    Back_but = tk.Button(simu_frame, text="Main Page", fg="green", command=back_main3)
    Back_but.place(relx=0.5, rely=0.7, anchor="center")





def open_welcome():
    global welcome_frame
    global logo_photo
    # Create Welcome page
    welcome_frame = tk.Frame(window)
    welcome_frame.pack(fill="both", expand=True)

    # 
    tk.Label(welcome_frame).pack()

    # Create Welcome labels
    #logo_image = Image.open("BCI.png")  # Replace with your logo image path
    #logo_photo = ImageTk.PhotoImage(logo_image)
    # Create a Label with the logo image and place it in the welcome_frame
    #logo_label = tk.Label(welcome_frame, image=logo_photo)
    #logo_label.place(relx=0.5, rely=0.2, anchor="center")

  
    original_image = Image.open("resources/BCI.png")  

    #resized_image = original_image.resize((200, 200), Image.ANTIALIAS)
    original_image.thumbnail((200, 200))

    photo = ImageTk.PhotoImage(original_image)
    image_label = tk.Label(welcome_frame, image=photo)
    image_label.image = photo 
    image_label.pack()

    welcome_label = tk.Label(welcome_frame, text="Main Menu", font=("Arial", 16, "bold"))
    welcome_label.place(relx=0.5, rely=0.3, anchor="center")
    # Create Version label
    version_label = tk.Label(window, text="Version 0.0.2.6", fg="gray")
    version_label.place(relx=1.0, rely=1.0, anchor="se")
    
    # Create Start button
    modelfit_button = tk.Button(welcome_frame, text="Model Fitting", command=open_modelfit, width=30, height=3)
    modelfit_button.place(relx=0.5, rely=0.4, anchor="center")

    simu_button = tk.Button(welcome_frame, text="Sensory Simulation", command=open_simu, width=30, height=3)
    simu_button.place(relx=0.5, rely=0.5, anchor="center")


    about_button = tk.Button(welcome_frame, text="About BCI Toolbox", command=about_bci, width=15, height=2)
    about_button.place(relx=0.5, rely=0.8, anchor="center")

    # 
    tk.Label(welcome_frame).pack()
