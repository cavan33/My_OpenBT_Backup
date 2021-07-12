# Designed to go inside the OPENBT class
def mopareto(self, cmdopt = 'serial', fit1=None, fit2=None, fit3=None, q_lower=0.025, q_upper=0.975, tc = 4):  
        """ Pareto Front Multiobjective Optimization using 2 fitted BART models
        """
        # params
        if(fit1 == None or fit2 == None): sys.exit("No fitted models specified!\n")
        # if(fit1['xi'][0] != fit2['xi'][0]): sys.exit("Models not compatible\n") # Checking xicuts
        if(fit1['ndpost'] != fit2['ndpost']): sys.exit("Models have different number of posterior samples\n")
        p = len(fit1['xi'])
        d = 2 + int(fit3 != None)
        print("d = ", d)
        m1 = fit1['ntree']
        m2 = fit2['ntree']
        mh1 = fit1['ntreeh']
        mh2 = fit2['ntreeh']
        m3 = 0; mh3 = 0
        if(fit3 != None):
           if(fit3['ndpost'] != fit2['ndpost']): sys.exit("Models have different number of posterior samples\n")
           m3 = fit3['ntree']
           mh3 = fit3['ntreeh']
        else:
            fit3={}
            fit3['modelname'] = "null"
            fit3['fpath'] = "null"
            fit3['fmean'] = 0.
        
        nd = fit1['ndpost']
        modelname = fit1['modelname']
        
        # Write to config file:
        # I dropped the if statement about fit3 (from R source code), b/c it's unnecessary
        mopareto_params = [fit1['modelname'], fit2['modelname'], fit3['modelname'],
                    fit1['xiroot'], fit2['fpath'], fit3['fpath'], nd, m1, mh1, m2, 
                    mh2, m3, mh3, p, fit1['minx'], fit1['maxx'], fit1['fmean'], 
                    fit2['fmean'], fit3['fmean'], tc]
        self.configfile = Path(self.fpath / "config.mopareto")
        # print("Directory for mopareto:", self.fpath) # print(sobol_params); 
        with self.configfile.open("w") as tfile:
            for param in mopareto_params:
                if type(param) != str and type(param) != int: # Makes minx & maxx into writable quantities, not arrays
                     for item in param:
                          tfile.write(str(item)+"\n")
                else: tfile.write(str(param)+"\n")
        # Run Pareto Front program: optional to use MPI.
        # Run sobol program
        cmd = "openbtmopareto"
        if(cmdopt == 'serial'):
             sp = subprocess.run([cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        elif(cmdopt == 'MPI'):
             sp = subprocess.run(["mpirun", "-np", str(tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        else:
             sys.exit('MoPareto: Invalid cmdopt (command option)')
        # print(sp)
        # Read in result (and set a bunch of extra attributes?):
        ii = 1; u = 0  # to modify temp indexing if fit3 exists
        """ IDK how to do this, so this is still the R version:
        for(i in 1:tc) 
        {
           con=file(paste(fit1$folder,"/",fit1$modelname,".mopareto",i-1,sep=""))
           open(con)
           s=readLines(con,1)
           while(length(s)>0) {
              temp=as.numeric(unlist(strsplit(s," ")))
              k=as.integer(temp[1])
              theta=matrix(0,ncol=k,nrow=d)
              theta[1,]=temp[2:(2+k-1)]
              theta[2,]=temp[(2+k):(2+2*k-1)]
              if(!is.null(fit3)) { 
                 theta[3,]=temp[(2+2*k):(2+3*k-1)] 
                 u=k
              }
              a=matrix(0,nrow=p,ncol=k)
              for(i in 1:p) a[i,]=temp[(2+(2+i-1)*k):(2+(2+i)*k-1)+u]
              b=matrix(0,nrow=p,ncol=k)
              for(i in 1:p) b[i,]=temp[(2+(2+p+i-1)*k):(2+(2+p+i)*k-1)+u]
              entry=list()
              entry[["theta"]]=theta
              entry[["a"]]=a
              entry[["b"]]=b
              res[[ii]]=entry
              ii=ii+1
              s=readLines(con,1)
           }
           close(con)
        }
        
        mopareto_files = sorted(list(self.fpath.glob(fit1['modelname'], ".mopareto*")))
        # print(mopareto_files)
        s = np.loadtxt(mopareto_files[0])
        for i in range(1, tc):
             read = open(mopareto_files[i], "r"); lines = read.readlines()
             if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                  s = np.vstack((s, np.loadtxt(mopareto_files[i])))   
        # print(s.shape); print(s[0:10])
        labs_temp = list(itertools.combinations(range(1, p + 1), 2))
        labs = np.empty(len(labs_temp), dtype = '<U4')
        for i in range(len(labs_temp)):
             labs[i] =  ', '.join(map(str, labs_temp[i]))
        # print(s.shape)
        ncol = s.shape[1]; # nrow = s.shape[0]
        # All this is the same as R, but the beginning of the indices are shifted by
        # 1 since Python starts at index 0. Remember, Python omits the column at the 
        # end of the index, so the end index is actually the same as R!
        self.num_pairs = int(self.p*(self.p-1)/2)
        res = {}
        return res
"""
