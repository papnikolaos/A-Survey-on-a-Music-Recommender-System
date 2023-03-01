from tkinter import *
from tkinter import ttk
from tkinter import Canvas
from tkinter import Frame
from tkinter import StringVar
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PrepareData import GetData
import threading
from BaselineSurpriseModels import Baseline_Models
from songs_stats import *
from AutoRec import *
from NeuralImplementations import LightGCN_Implementation,NCF_Implementation
from RBM_Implemantation import RBM_Implementation
from time import time

def reset():
    selected_models.clear()
    selected_models.clear()
    selected_models_predictions.clear()
    baseline_constructors.clear()
    users_to_predict.clear()

# Pop-up window for model performance
def command1(model_num):
    reset()
    selected_models.append(model_num)
    b = Baseline_Models(selected_models,"pipeline/train.txt","pipeline/test.txt")
    baseline_constructors.append(b)
    x = b.train_models()

    top_results = Toplevel()
    top_results.geometry("620x100")
    top_results.title("Model Results")
    Label(top_results,text = "Model").grid(row=0,column=0)
    Label(top_results,text = "RMSE").grid(row=0,column=1)
    Label(top_results,text = "MSE").grid(row=0,column=2)
    Label(top_results,text = "MAE").grid(row=0,column=3)
    Label(top_results,text = "FCP").grid(row=0,column=4)
    Label(top_results,text = "Precision").grid(row=0,column=5)
    Label(top_results,text = "Recall").grid(row=0,column=6)
    Label(top_results,text = "Training Time (s)").grid(row=0,column=7)
    (rmse,mse,mae,pr,re,fcp,time) = next(x)
    Label(top_results,text = models[selected_models[0]-1]).grid(row=1,column=0)
    Label(top_results,text = str(round(rmse,4))).grid(row=1,column=1)
    Label(top_results,text = str(round(mse,4))).grid(row=1,column=2)
    Label(top_results,text = str(round(mae,4))).grid(row=1,column=3)
    Label(top_results,text = str(round(fcp,4))).grid(row=1,column=4)
    Label(top_results,text = str(round(pr,4))).grid(row=1,column=5)
    Label(top_results,text = str(round(re,4))).grid(row=1,column=6)
    Label(top_results,text = str(round(time,4))).grid(row=1,column=7)
    button = Button(top_results, text= "Make recommendations", command = partial(command4,top_results))
    button.grid(row=2,column=7,sticky=E)
    top_results.mainloop()

#Pop-up window for parameters set up
def command2():
    users = int(entry1.get())
    songs = int(entry2.get())
    global user_num
    global song_num
    song_num = songs
    user_num = users
    top.destroy()

    top_prog = Toplevel(main_window)
    top_prog.geometry("210x100")
    top_prog.title("Parameters Set Up")

    Label(top_prog,text = "Preparing the data...",height=2,width=20).grid(row=0,column=0)
    progress = ttk.Progressbar(top_prog, orient=HORIZONTAL,length=190, mode='indeterminate')
    progress.grid(row=1,column=0)

    t = threading.Thread(target=command2_thread,args=(train_file_path,test_file_path,users,songs,progress,top_prog,))
    t.start()
    progress.start(10)
    top_prog.mainloop()   

#Thread for data preparation
def command2_thread(train_file_path,test_file_path,users,songs,progress,top_prog):
    p = GetData(train_file_path,test_file_path,users,songs)
    p.create_filtered_dataset()
    progress.stop()
    top_prog.destroy()

#Menu to select surprise models
def command3():
    top = Toplevel(main_window)
    top.geometry("200x270")
    top.title("Model Selection")
    agreement = []

    for i in range(11):
        var = StringVar()
        checkbox =ttk.Checkbutton(top,
                text=models[i],
                command=None,
                variable=var,
                onvalue='Yes',
                offvalue='No')
        checkbox.grid(row=i,column=0)
        agreement.append(var)
    button = Button(top, text= "OK", command = partial(command3_models,agreement,top),height=1,width=4)
    button.grid(row=12,sticky=E)
    top.mainloop()

#Pop-up wondow for performance of selected surprise models
def command3_models(agreement,top): 
    reset()  
    for i in range(11):
        if(agreement[i].get() == 'Yes'):
            selected_models.append(i+1)
    top.destroy()
    b = Baseline_Models(selected_models,"pipeline/train.txt","pipeline/test.txt")
    baseline_constructors.append(b)
    x = b.train_models()

    top_results = Toplevel()
    top_results.geometry("620x300")
    top_results.title("Model Results")
    Label(top_results,text = "Model").grid(row=0,column=0)
    Label(top_results,text = "RMSE").grid(row=0,column=1)
    Label(top_results,text = "MSE").grid(row=0,column=2)
    Label(top_results,text = "MAE").grid(row=0,column=3)
    Label(top_results,text = "FCP").grid(row=0,column=4)
    Label(top_results,text = "Precision").grid(row=0,column=5)
    Label(top_results,text = "Recall").grid(row=0,column=6)
    Label(top_results,text = "Training Time (s)").grid(row=0,column=7)

    for i in range(len(selected_models)):
        (rmse,mse,mae,pr,re,fcp,time) = next(x)
        Label(top_results,text = models[selected_models[i]-1]).grid(row=i+1,column=0)
        Label(top_results,text = str(round(rmse,4))).grid(row=i+1,column=1)
        Label(top_results,text = str(round(mse,4))).grid(row=i+1,column=2)
        Label(top_results,text = str(round(mae,4))).grid(row=i+1,column=3)
        Label(top_results,text = str(round(fcp,4))).grid(row=i+1,column=4)
        Label(top_results,text = str(round(pr,4))).grid(row=i+1,column=5)
        Label(top_results,text = str(round(re,4))).grid(row=i+1,column=6)
        Label(top_results,text = str(round(time,4))).grid(row=i+1,column=7)

    button = Button(top_results, text = "Make recommendations", command = partial(command4,top_results))
    button.grid(row=len(selected_models)+1,column=7,sticky=E)    
    top_results.mainloop()

#Pop-up window for recommendation request
def command4(top_results):
    top_results.destroy()
    top_preds = Toplevel()
    top_preds.geometry("620x300")
    top_preds.title("Model Results")
    Label(top_preds,text = 'Users to make recommendations (separated with ,): ').grid(row=0)
    entry = Entry(top_preds, width=30)
    entry.grid(row=1)

    Label(top_preds,text = 'Number of recommendations: ').grid(row=2)
    entry_rec_num = Entry(top_preds,width=10)
    entry_rec_num.grid(row=3)

    agreement = []
    for i in range(len(selected_models)):
        var = StringVar()
        checkbox =ttk.Checkbutton(top_preds,
                text=models[selected_models[i]-1],
                command=None,
                variable=var,
                onvalue='Yes',
                offvalue='No')
        checkbox.grid(row=i+4)
        agreement.append(var)

    button = Button(top_preds, text= "OK", command = partial(command4_models,agreement,entry,entry_rec_num,top_preds),height=1,width=4)
    button.grid(row=len(selected_models)+4)
    top_preds.mainloop()

def onFrameConfigure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))

#Pop-up window for top recommendations
def command4_models(agreement,entry,entry_rec_num,top_preds):
    users_lst = entry.get().split(',')
    rec_num = int(entry_rec_num.get())
    for i in range(len(users_lst)):
        users_to_predict.append(int(users_lst[i]))
    for i in range(len(selected_models)):
        if(agreement[i].get() == 'Yes'):
            selected_models_predictions.append([i,models[selected_models[i]-1]])
    top_preds.destroy()
    
    top_preds_summary = Toplevel()
    top_preds_summary.geometry("620x300")
    top_preds_summary.title("Top predictions")

    text = ""
    for user in users_to_predict: 
        for m in selected_models_predictions:
            text += "\nModel used: %s"%(m[1])
            text += baseline_constructors[0].make_recommendations(user,rec_num,m[0])        

    canvas = Canvas(top_preds_summary, borderwidth=0)
    frame = Frame(canvas)
    vsb = Scrollbar(top_preds_summary, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((4,4), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    Label(frame,text=text).grid(row=0,column=0)
    top_preds_summary.mainloop()

def command5_top_songs():
    best_songs, best_song_ratings = top_songs(song_num)
    fig, ax = plt.subplots(figsize=(11,6.6))
    bars = ax.bar(best_songs, best_song_ratings,color='mediumpurple')
    ax.set_title('Top rated songs')
    ax.bar_label(bars)

    canvas = FigureCanvasTkAgg(fig,master=main_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 0,column = 0,sticky=E+W+N+S)

    button = Button(main_window, text= "Close", command= partial(command_close,canvas),height=2,width=8)
    button.grid(row = 1,column = 0,sticky=NS)

def command_close(canvas):
    for widgets in main_window.winfo_children():
        if type(widgets) == Button:              
            widgets.destroy()
    for item in canvas.get_tk_widget().find_all():
       canvas.get_tk_widget().delete(item)
    canvas.get_tk_widget().destroy()

def command5_top_artists():
    best_artists, best_artist_ratings = top_artists(song_num)
    fig, ax = plt.subplots(figsize=(11,6.6))
    bars = ax.bar(best_artists, best_artist_ratings,color='mediumpurple')
    ax.set_title('Top rated artists')
    ax.bar_label(bars)

    canvas = FigureCanvasTkAgg(fig,master=main_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 0,column = 0,sticky=E+W+N+S)

    button = Button(main_window, text= "Close", command= partial(command_close,canvas),height=2,width=8)
    button.grid(row = 1,column = 0,sticky=NS)

def command5_top_albums():
    best_albums, best_album_ratings = top_albums(song_num)
    fig, ax = plt.subplots(figsize=(11,6.6))
    bars = ax.bar(best_albums, best_album_ratings,color='mediumpurple')
    ax.set_title('Top rated albums')
    ax.bar_label(bars)

    canvas = FigureCanvasTkAgg(fig,master=main_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 0,column = 0,sticky=E+W+N+S)

    button = Button(main_window, text= "Close", command= partial(command_close,canvas),height=2,width=8)
    button.grid(row = 1,column = 0,sticky=NS)

def command5(param):    
    if(param == 1):
        command5_top_songs()               
    elif(param == 2):
        command5_top_artists()
    elif(param == 3):
        command5_top_albums()

#Pop-up window for neural net performance
def command6(model):
    top_results = Toplevel()
    top_results.geometry("620x120")
    top_results.title("Model Results")
    Label(top_results,text = "Model").grid(row=0,column=0)
    Label(top_results,text = "RMSE").grid(row=0,column=1)
    Label(top_results,text = "Precision").grid(row=0,column=2)
    Label(top_results,text = "Recall").grid(row=0,column=3)
    Label(top_results,text = "Training Time (s)").grid(row=0,column=4)
    performance = [0]*4

    if(model == 1):
        Label(top_results,text = 'AutoRec').grid(row=1,column=0)
        r, r_mask = get_data(user_num,song_num)
        train_r, test_r, train_mask, test_mask = train_test_split(r,r_mask,test_size=0.25,shuffle=True)
        torch_dataset = Data.TensorDataset(torch.from_numpy(train_r),torch.from_numpy(train_mask),torch.from_numpy(train_r))
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=10
        )
        rec = Autorec(user_num,song_num,50,1e-2)
        optimizer = optim.Adam(rec.parameters(), lr = 1e-3, weight_decay=1e-4)
        tic = time()
        for i in range(100):
            train(i,loader,rec,optimizer,train_mask)
        toc = time()
        performance[3] = toc - tic
        (performance[0],performance[1],performance[2]) = test(user_num,song_num,rec,test_r,test_mask)
        p = [r,rec,r_mask]
                 
    elif(model == 2):
        Label(top_results,text = 'NCF').grid(row=1,column=0)
        p = NCF_Implementation("pipeline/train.txt","pipeline/test.txt",user_num,song_num)
        p.train()
        performance[:] = p.evaluate()
        
    elif(model == 3):
        Label(top_results,text = 'LightGCN').grid(row=1,column=0)
        p = LightGCN_Implementation(user_num)
        performance[3] = p.trainGCN()
        performance[0:2] = p.evaluate()

    elif(model == 4):
        Label(top_results,text = 'RBM').grid(row=1,column=0)
        p = RBM_Implementation(user_num)
        performance[3] = p.train()
        (performance[0],performance[1],performance[2]) = p.evaluate()
        
    Label(top_results,text = str(round(performance[0],4))).grid(row=1,column=1)
    Label(top_results,text = str(round(performance[1],4))).grid(row=1,column=2)
    Label(top_results,text = str(round(performance[2],4))).grid(row=1,column=3)
    Label(top_results,text = str(round(performance[3],4))).grid(row=1,column=4)
    Label(top_results,text = "Number of recommendations: ").grid(row=2,column=0)
    entry_rec = Entry(top_results,width=4)
    entry_rec.grid(row=2,column=1)
    Label(top_results,text = 'Users to make recommendations (separated with ,): ').grid(row=3,column=0)
    entry_users = Entry(top_results,width=10)
    entry_users.grid(row=3,column=1)
    button = Button(top_results, text= "Make recommendations", command = partial(command7,top_results,p,entry_rec,\
        entry_users))
    button.grid(row=4,column=4,sticky=E)
    top_results.mainloop()

#Pop-up window for top recommendations of neural nets
def command7(top_results,p,entry_rec,entry_users):
    text = ""
    recommendations_num = int(entry_rec.get())
    users_lst = [int(i) for i in entry_users.get().split(',')]
    for user in users_lst:
        if not isinstance(p, list):
            text += p.make_recommendations(user,recommendations_num)
        else:
            text += make_recommendations_autorec(user,p,recommendations_num)
    top_results.destroy()

    top_preds_summary = Toplevel()
    top_preds_summary.geometry("620x300")
    top_preds_summary.title("Top predictions")      

    canvas = Canvas(top_preds_summary, borderwidth=0)
    frame = Frame(canvas)
    vsb = Scrollbar(top_preds_summary, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((4,4), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    Label(frame,text=text).grid(row=0,column=0)
    top_preds_summary.mainloop()


models = ['Normal Predictor','Baseline Estimate','Basic k-NN','k-NN with Means','k-NN with z-score',\
        'k-NN with Baseline Rating','MF with SVD','MF with SVD++','Non-Negative MF','Slope One','Co-Clustering']

selected_models = []
selected_models_predictions = []
baseline_constructors = []
users_to_predict = []

train_file_path = "Dataset/train_0.txt"
test_file_path = "Dataset/test_0.txt"

main_window = Tk()
main_window.title("Music Recommender System")
main_window.configure(background="black")

photo = PhotoImage(file='music_img.png')
main_window.geometry("{width}x{height}".format(width = photo.width(),height = photo.height()))
main_window.resizable(False,False)

Label(main_window,image=photo,bg="black").place(relheight=1,relwidth=1)

my_menu = Menu(main_window)
main_window.config(menu=my_menu)

surprise_menu = Menu(my_menu)
my_menu.add_cascade(label='Surprise Models',menu=surprise_menu)

basic_algo_menu = Menu(surprise_menu)
surprise_menu.add_cascade(label='Basic Algorithms',menu=basic_algo_menu)
basic_algo_menu.add_command(label='Normal Predictor',command=partial(command1,1))
basic_algo_menu.add_command(label='Baseline Estimate',command=partial(command1,2))

knn_algo_menu = Menu(surprise_menu)
surprise_menu.add_cascade(label='k-NN Algorithms',menu=knn_algo_menu)
knn_algo_menu.add_command(label='Basic k-NN',command=partial(command1,3))
knn_algo_menu.add_command(label='k-NN with Means',command=partial(command1,4))
knn_algo_menu.add_command(label='k-NN with z-score',command=partial(command1,5))
knn_algo_menu.add_command(label='k-NN with Baseline Rating',command=partial(command1,6))

mf_algo_menu = Menu(surprise_menu)
surprise_menu.add_cascade(label='MF Algorithms',menu=mf_algo_menu)
mf_algo_menu.add_command(label='MF with SVD',command=partial(command1,7))
mf_algo_menu.add_command(label='MF with SVD++',command=partial(command1,8))
mf_algo_menu.add_command(label='Non-Negative MF',command=partial(command1,9))

surprise_menu.add_command(label='Slope One',command=partial(command1,10))
surprise_menu.add_command(label='Co-Clustering',command=partial(command1,11))
surprise_menu.add_separator()
surprise_menu.add_command(label='Combination of models',command= command3)

nn_menu = Menu(my_menu)
my_menu.add_cascade(label='Neural Net Models',menu=nn_menu)
nn_menu.add_command(label='AutoRec',command=partial(command6,1))
nn_menu.add_command(label='NCF',command=partial(command6,2))
nn_menu.add_command(label='LightGCN',command=partial(command6,3))
nn_menu.add_command(label='RBM',command=partial(command6,4))

stats_menu = Menu(my_menu)
my_menu.add_cascade(label='Statistics - Plots',menu=stats_menu)
stats_menu.add_command(label='Top rated songs',command = partial(command5,1))
stats_menu.add_command(label='Top rated artists',command = partial(command5,2))
stats_menu.add_command(label='Top rated albums',command = partial(command5,3))

exit_menu = Menu(my_menu)
my_menu.add_cascade(label='Exit',menu=exit_menu)
exit_menu.add_command(label='Exit app...',command=partial(exit,0))

top = Toplevel(main_window)
top.geometry("250x100")
top.title("Parameters Set Up")

Label(top,text = "Number of users: ",height=2,width=20).grid(row=0,column=0,sticky=E+W+N+S)
entry1 = Entry(top, width= 10)
entry1.grid(row=0,column=1)
Label(top,text = "Number of songs: ",height=2,width=20).grid(row=1,column=0,sticky=E+W+N+S)
entry2 = Entry(top, width= 10)
entry2.grid(row=1,column=1)
button = Button(top, text= "OK", command= command2,height=1,width=4)
button.grid(row=2,sticky=E)
top.wm_transient(main_window)
main_window.mainloop()