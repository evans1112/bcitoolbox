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
from scipy.fftpack import dct
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from .BCIbox import simulateVV
from .BCIbox import prod_gaus
from .BCIbox import prod3gauss
from .BCIbox import plotKonrads

from tkinter import filedialog
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.stats import norm
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def gui():
    global window, data, a, b, c, result_label, Status_label, progressbar, count, bounds
    data = None
    a = None
    b = None
    c = None
    result_label = None
    Status_label = None
    progressbar = None
    count = 5
    bounds = [(0, 1),(0.1, 3),(0.1, 3),(0.1, 3),(0, 3.5)]
    
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
    # Run
    window.mainloop()
    
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
    global count
    global bounds
    checkbox_vars = [var1, var2, var3, var4, var5, var6, var7]
    count = sum(var.get() for var in checkbox_vars)
    print("Number of free parameters:", count)
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
    bounds = bounds[:count]
    
    parameters_window.destroy()
    
def save_setting():
    global count
    global bounds
    checkbox_vars = [var1, var2, var3, var4, var5, var6, var7]
    count = sum(var.get() for var in checkbox_vars)
    print("Number of free parameters:", count)
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
    bounds = bounds[:count]
    
    
def open_parameters_window():
    global parameters_window
    global var1, var2, var3, var4, var5, var6, var7
    global entry1_l, entry1_u, entry2_l, entry2_u, entry3_l, entry3_u, entry4_l, entry4_u, entry5_l, entry5_u, entry6_l, entry6_u, entry7_l, entry7_u
    # 
    parameters_window = tk.Toplevel(window)
    parameters_window.title("Parameters")
    parameters_window.geometry("400x300")
    
    paras_label = tk.Label(parameters_window, text="Parameters", fg="blue")
    paras_label.place(relx=0.1, rely=0.02, anchor="w")
    low_label = tk.Label(parameters_window, text="Lower bound", fg="blue")
    low_label.place(relx=0.5, rely=0.02, anchor="center")
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
    
    # 
    entry1_l = tk.Entry(parameters_window, width=5)
    entry2_l = tk.Entry(parameters_window, width=5)
    entry3_l = tk.Entry(parameters_window, width=5)
    entry4_l = tk.Entry(parameters_window, width=5)
    entry5_l = tk.Entry(parameters_window, width=5)
    entry6_l = tk.Entry(parameters_window, width=5)
    entry7_l = tk.Entry(parameters_window, width=5)

    # 
    entry1_l.insert(tk.END, 0)
    entry2_l.insert(tk.END, 0.1)
    entry3_l.insert(tk.END, 0.1)
    entry4_l.insert(tk.END, 0.1)
    entry5_l.insert(tk.END, 0)
    entry6_l.insert(tk.END, 1)
    entry7_l.insert(tk.END, 1)

    entry1_l.place(relx=0.5, rely=0.1, anchor="center")
    entry2_l.place(relx=0.5, rely=0.2, anchor="center")
    entry3_l.place(relx=0.5, rely=0.3, anchor="center")
    entry4_l.place(relx=0.5, rely=0.4, anchor="center")
    entry5_l.place(relx=0.5, rely=0.5, anchor="center")
    entry6_l.place(relx=0.5, rely=0.6, anchor="center")
    entry7_l.place(relx=0.5, rely=0.7, anchor="center")
    

    # 
    entry1_u = tk.Entry(parameters_window, width=5)
    entry2_u = tk.Entry(parameters_window, width=5)
    entry3_u = tk.Entry(parameters_window, width=5)
    entry4_u = tk.Entry(parameters_window, width=5)
    entry5_u = tk.Entry(parameters_window, width=5)
    entry6_u = tk.Entry(parameters_window, width=5)
    entry7_u = tk.Entry(parameters_window, width=5)
    
    entry1_u.insert(tk.END, 1)
    entry2_u.insert(tk.END, 3)
    entry3_u.insert(tk.END, 3)
    entry4_u.insert(tk.END, 3)
    entry5_u.insert(tk.END, 3.5)
    entry6_u.insert(tk.END, 5)
    entry7_u.insert(tk.END, 5)

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
    # 用户选择的图片序号
    selected_image = tk.IntVar()
    if ifsave==2:
        target_folder = filedialog.askdirectory()
    else:
        num_plot = tk.simpledialog.askinteger("Plot", "Please enter the position number of data you want to plot (e.g. 1 for the first one)", minvalue=0, maxvalue=10000)

    for ndata in range (len(data_matrices)):
        plt.figure()
        [error,modelprop,dataprop,d] = simulateVV(parameters_list[ndata], 10000, data_matrices[ndata], 1, strategy = strategy_list[ndata], fitType = 'mll')
        a = dataprop.shape
        condi = int((a[1] + 1) ** 0.5)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  

        for i in range(a[1]):
            if condi < 11:
                plt.subplot(condi, condi, i + 2)
            else:
                plt.figure(int(i/condi) + 1)
                plt.subplot(condi, 1, (i % condi) + 1)
        
            plt.plot(dataprop[0, i, :], 'b')
     
            plt.plot(dataprop[1, i, :], 'r')
            plt.axis([0, a[2]-1, 0, 1])

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
        file_path_simudata = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path_simudata:
            with open(file_path_simudata, 'w') as file:
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
        plt.annotate(f'{x1[peak_index_D1]:.2f}', xy=(x1[peak_index_D1], peak_value_D1), xytext=(x1[peak_index_D1], peak_value_D1 + 0.05),
             arrowprops=dict(arrowstyle='-', color='black'))
        plt.annotate(f'{x2[peak_index_D2]:.2f}', xy=(x2[peak_index_D2], peak_value_D2), xytext=(x2[peak_index_D2], peak_value_D2 + 0.05),
             arrowprops=dict(arrowstyle='-', color='black'))

    if var_disp.get() and var_mean.get() and var_respD.get():
        plt.annotate(f'{mean_D1:.2f}', xy=(mean_D1, kde1.evaluate(mean_D1)), xytext=(mean_D1, kde1.evaluate(mean_D1) - 0.05),
             arrowprops=dict(arrowstyle='-', color='black'))
        plt.annotate(f'{mean_D2:.2f}', xy=(mean_D2, kde2.evaluate(mean_D2)), xytext=(mean_D2, kde2.evaluate(mean_D2) - 0.05),
             arrowprops=dict(arrowstyle='-', color='black'))

            






    plt.legend(prop={'size': 8})
    plt.xlabel('Value')
    plt.ylabel('Probability')
    
    root_sim = tk.Tk()
    root_sim.title('Figure')
    label_sim = tk.Label(root_sim, text='Simulation Result')
    label_sim.pack()

    canvas_sim = FigureCanvasTkAgg(plt.gcf(), master=root_sim)
    canvas_sim.draw()
    canvas_sim.get_tk_widget().pack()

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
        header = "File Name\t" + "\t".join(header_list[:count]) + "\tError\tStrategy\n"
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
        strategies.append(('sel', 'Selecting'))
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


    np.random.seed(13)
    parameters_list = []
    error_list = []
    strategy_list = []
    finalR = ""
    x0 = np.random.rand(count)
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

        for strategy, strategy_name in strategies:
            # Show "Please wait..."
            result_label.config(text=f"Fitting with {strategy_name} strategy & {fit_t} fit type for data {ndata+1}...", fg="blue")
            window.update()
        
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

            result = minimize(lambda paras: simulateVV(paras, nseeds, data, 0, strategy=strategy, fitType=fit_t)[0],
                          x0, method='Powell', bounds=bounds, callback=minimize_callback)

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Running time for {strategy_name} strategy:", execution_time, "s")

            if result.success and result.fun < best_error:
                best_error = result.fun
                best_result = result
                best_strategy = strategy_name
                plot_strategy = strategy

        if best_result is not None:
            # Get result2.x & result2.fun
            result_x = str(best_result.x)
            result_fun = str(best_result.fun)

            # Show "Result is ... (fun)"
            text = f"Best result for data {ndata+1} is {result_x}\nError is {result_fun}\nStrategy: {best_strategy}"
            finalR = finalR + '\n' + text
        else:
            result_label.config(text="No valid results found.", fg="red")

        # Hide bar
        progressbar.pack_forget()
        parameters_list.append(best_result.x)
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
    global var_a
    global var_b
    global var_c
    global var_mll
    global var_mr2
    global var_sse
    global result_label
    global entry_seeds
    global fit_frame
    # Create homepage framework
    fit_frame = tk.Frame(window)
    welcome_frame.pack_forget()  # Hide Welcome
    fit_frame.pack()  # Show Homepage
    # Create Text Box
    entry = tk.Entry(fit_frame, width=60)
    entry.pack()

    button = tk.Button(fit_frame, text="Import", command=lambda: import_csv(entry))
    button.place(relx=0.4, rely=0.15, anchor="center")
    button2 = tk.Button(fit_frame, text="Open file", command=lambda: open_file(entry))
    button2.place(relx=0.6, rely=0.15, anchor="center")
    question_button = tk.Button(fit_frame, text="?")
    question_button.place(relx=0.9, rely=0.15, anchor="center")
    
    seeds_label = tk.Label(fit_frame, text="Number of simulations : ", fg="black")
    seeds_label.place(relx=0.4, rely=0.235, anchor="center")
    
    entry_seeds = tk.Entry(fit_frame, width = 5)
    entry_seeds.insert(tk.END, 1000)
    entry_seeds.place(relx=0.6, rely=0.235, anchor="center")

    # Create checkboxes and corresponding functions
    var_a = tk.BooleanVar()
    var_b = tk.BooleanVar()
    var_c = tk.BooleanVar()

    check_button_a = tk.Checkbutton(fit_frame, text="Averaging", variable=var_a, command=function_a)
    check_button_b = tk.Checkbutton(fit_frame, text="Selecting", variable=var_b, command=function_b)
    check_button_c = tk.Checkbutton(fit_frame, text="Matching", variable=var_c, command=function_c)

    strategy_label = tk.Label(fit_frame, text="select at least one strategy", fg="dark green")
    strategy_label.place(relx=0.7, rely=0.3, anchor="center")
    check_button_a.place(relx=0.7, rely=0.4, anchor="center")
    check_button_b.place(relx=0.7, rely=0.5, anchor="center")
    check_button_c.place(relx=0.7, rely=0.6, anchor="center")

    # Create checkboxes and corresponding functions
    var_mll = tk.BooleanVar(value=True)
    var_mr2 = tk.BooleanVar()
    var_sse = tk.BooleanVar()

    check_button_mll = tk.Checkbutton(fit_frame, text="mll", variable=var_mll)
    check_button_mr2 = tk.Checkbutton(fit_frame, text="mr2", variable=var_mr2)
    check_button_sse = tk.Checkbutton(fit_frame, text="sse", variable=var_sse)

    type_label = tk.Label(fit_frame, text="select one fit type", fg="dark green")
    type_label.place(relx=0.3, rely=0.3, anchor="center")
    check_button_mll.place(relx=0.3, rely=0.4, anchor="center")
    check_button_mr2.place(relx=0.3, rely=0.5, anchor="center")
    check_button_sse.place(relx=0.3, rely=0.6, anchor="center")

    placeholder = tk.Frame(fit_frame, height=180)
    placeholder.pack()
    
    parameters_button = tk.Button(fit_frame, text="Parameters", command=open_parameters_window)
    parameters_button.pack()
    
    run_button = tk.Button(fit_frame, text="Run", command=run_code)
    run_button.pack()
    # Show results' labels
    result_label = tk.Label(fit_frame, text= "Result will be here...: ", fg="gray")
    result_label.pack()
    
    # Plot
    plot_button = tk.Button(fit_frame, text="Plot", command=lambda: plot_func(1))
    plot_button.pack()

    # Save
    save_fit = tk.Button(fit_frame, text="Save", command=lambda: plot_func(2))
    save_fit.pack()

    
    # Back
    Back_but = tk.Button(fit_frame, text="Main Page", fg="green", command=back_main1)
    Back_but.pack()

    
    
def back_main():
    simu_frame.pack_forget()
    open_welcome()
    
def back_main1():
    fit_frame.pack_forget()
    open_welcome()
    

def update_value(value):
      
    entry_pcommon.insert(variable_pcommon)
    entry_sigmaU.insert(variable_sigmaU)
    entry_sigmaD.insert(variable_sigmaD)
    entry_sigmaZ.insert(variable_sigmaZ)
    entry_sti1.insert(variable_sti1)
    entry_sti2.insert(variable_sti2)

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
    
    
    

    # 创建滑轨部件，设置范围和回调函数
    scale_pcommon = tk.Scale(simu_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_pcommon, command=update_value, length=500)
    scale_pcommon.pack()
    
    entry_pcommon = tk.Entry(simu_frame, textvariable=variable_pcommon, width=5)
    entry_pcommon.pack()
    entry_pcommon.bind('<Return>', entry_changed)
    
    scale_sigmaU = tk.Scale(simu_frame, from_=0.1, to=3.5, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaU, command=update_value, length=500)
    scale_sigmaU.pack()
    
    entry_sigmaU = tk.Entry(simu_frame, textvariable=variable_sigmaU, width=5)
    entry_sigmaU.pack()
    entry_sigmaU.bind('<Return>', entry_changed)
    
    scale_sigmaD = tk.Scale(simu_frame, from_=0.1, to=3.5, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaD, command=update_value, length=500)
    scale_sigmaD.pack()
    
    entry_sigmaD = tk.Entry(simu_frame, textvariable=variable_sigmaD, width=5)
    entry_sigmaD.pack()
    entry_sigmaD.bind('<Return>', entry_changed)
    
    scale_sigmaZ = tk.Scale(simu_frame, from_=0.1, to=3.5, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_sigmaZ, command=update_value, length=500)
    scale_sigmaZ.pack()
    
    entry_sigmaZ = tk.Entry(simu_frame, textvariable=variable_sigmaZ, width=5)
    entry_sigmaZ.pack()
    entry_sigmaZ.bind('<Return>', entry_changed)
    
    scale_mup = tk.Scale(simu_frame, from_=0.1, to=3, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=variable_mup, command=update_value, length=500)
    scale_mup.pack()
    
    entry_mup = tk.Entry(simu_frame, textvariable=variable_mup, width=5)
    entry_mup.pack()
    entry_mup.bind('<Return>', entry_changed)
    if singleif==1:
        scale_sti1 = tk.Scale(simu_frame, from_=0, to=4, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=variable_sti1, command=update_value, length=500)
        scale_sti1.pack()
    
        entry_sti1 = tk.Entry(simu_frame, textvariable=variable_sti1, width=5)
        entry_sti1.pack()
        entry_sti1.bind('<Return>', entry_changed)
    
        scale_sti2 = tk.Scale(simu_frame, from_=0, to=4, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=variable_sti2, command=update_value, length=500)
        scale_sti2.pack()
    
        entry_sti2 = tk.Entry(simu_frame, textvariable=variable_sti2, width=5)
        entry_sti2.pack()
        entry_sti2.bind('<Return>', entry_changed)
    
        
        pcommon_label = tk.Label(simu_frame, text="Pcommon", fg="black")
        pcommon_label.place(relx=0.3, rely=0.08, anchor="w")
    
        sigmaU_label = tk.Label(simu_frame, text="sigmaU", fg="blue")
        sigmaU_label.place(relx=0.3, rely=0.19, anchor="w")
    
        sigmaD_label = tk.Label(simu_frame, text="sigmaD", fg="red")
        sigmaD_label.place(relx=0.3, rely=0.29, anchor="w")
    
        sigmaZ_label = tk.Label(simu_frame, text="sigmaZ", fg="black")
        sigmaZ_label.place(relx=0.3, rely=0.395, anchor="w")
    
        mup_label = tk.Label(simu_frame, text="Z_center", fg="black")
        mup_label.place(relx=0.3, rely=0.5, anchor="w")
    
    
        sti1_label = tk.Label(simu_frame, text="stimulusU", fg="blue")
        sti1_label.place(relx=0.3, rely=0.6, anchor="w")
    
        sti2_label = tk.Label(simu_frame, text="stimulusD", fg="red")
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
                
    var_a_simu = tk.BooleanVar(value=True)
    var_b_simu = tk.BooleanVar()
    var_c_simu = tk.BooleanVar()

    check_button_a_simu = tk.Checkbutton(simu_frame, text="Averaging", variable=var_a_simu, command=function_a_simu)
    check_button_b_simu = tk.Checkbutton(simu_frame, text="Selecting", variable=var_b_simu, command=function_b_simu)
    check_button_c_simu = tk.Checkbutton(simu_frame, text="Matching", variable=var_c_simu, command=function_c_simu)
    
    check_button_a_simu.place(relx=0.2, rely=0.8, anchor="e")
    check_button_b_simu.place(relx=0.53, rely=0.8, anchor="e")
    check_button_c_simu.place(relx=0.86, rely=0.8, anchor="e")       
    # Back
    plot_Simu_but = tk.Button(simu_frame, text="Main Page", fg="green", command=back_main)
    plot_Simu_but.pack()
def open_welcome():
    global welcome_frame
    # Create Welcome page
    welcome_frame = tk.Frame(window)
    welcome_frame.pack(fill="both", expand=True)

    # 
    tk.Label(welcome_frame).pack()

    # Create Welcome labels
    welcome_label = tk.Label(welcome_frame, text="Welcome to the BCI Toolbox!")
    welcome_label.place(relx=0.5, rely=0.3, anchor="center")
    # Create Version label
    version_label = tk.Label(window, text="Version 0.0.1", fg="gray")
    version_label.place(relx=1.0, rely=1.0, anchor="se")
    
    # Create Start button
    startFit_button = tk.Button(welcome_frame, text="Start Fitting", command=fit_page)
    startFit_button.place(relx=0.5, rely=0.4, anchor="center")

    startSim_button = tk.Button(welcome_frame, text="Simulate for Continuous Condition", command=lambda: simu_page(1))
    startSim_button.place(relx=0.5, rely=0.45, anchor="center")

    startSimall_button = tk.Button(welcome_frame, text="Simulate for Numerosity Task", command=lambda: simu_page(2))
    startSimall_button.place(relx=0.5, rely=0.5, anchor="center")

    about_button = tk.Button(welcome_frame, text="About BCI Toolbox", command=about_bci)
    about_button.place(relx=0.5, rely=0.6, anchor="center")

    # 
    tk.Label(welcome_frame).pack()

