import numpy as np

def getDepression(dem, flow_direction, flow_direction_parent, cellSize):
   #Define variable
    pits = np.array(np.empty((np.shape(dem)[0],np.shape(dem)[1])))
    pits[:]=np.NaN
    pit_count = np.nansum(flow_direction == -1)
    edgepit_count = np.nansum(flow_direction == -2)
    pitID = np.int32(np.array(range(1,pit_count + 1), dtype = int)) # pit cellID
    #pit bottom cell index
    #pitCell = [pitCell for (pitCell, val) in enumerate(np.ravel(flow_direction, order = 'F')) if (val == -1)]
    pitCell = np.where(flow_direction == -1)
    edgePitId =np.int32(np.flip(np.array(range(-edgepit_count,0), dtype = int)))
    edgePitCell = np.where(flow_direction == -2)
    areaCount = np.zeros((pit_count,1),dtype = int) #integer number of cell
    spilloverElevation = np.zeros((pit_count,1),dtype = float)#meter
    vca = np.zeros((pit_count,1),dtype = float)#volume to contributing area ratio (hours)
    volume = np.zeros((pit_count,1),dtype = float)#cubic meters
    filledVolume = np.zeros((pit_count,1),dtype = float)#cubic meters
    cellOverflowInto = np.int32(np.zeros((pit_count,2),dtype = float))#cell index of overflow location
    
    #Pits must be identified in the pit matrix in order to return the 
    #correct pit ID that each pit flows into (if not, many of these pits will 
    #flow into yet unidentifed pits that have ID 0)
    
    #Fill out the pit ID image/matrix
    pits[~np.isnan(dem)] = 0 #if pits' element == NaN change the cell to 0 
    pits[pitCell] = pitID
    pits[edgePitCell] = edgePitId
    
    cellIndexes = [ [] for _ in range(pit_count)] #create a list within an list

    for p in range(0,np.shape(pitCell)[1]):
        pID = p + 1
        j = 0
        i = 0
        chunk = 50
        cellIndexes[p] = np.empty((chunk,2), dtype = object) 
        cellIndexes[p][0][0] = pitCell[0][p]
        cellIndexes[p][0][1] = pitCell[1][p]
        
        
        while i <= j:
            parent = flow_direction_parent[cellIndexes[p][i][0]][cellIndexes[p][i][1]]
            if len(parent) != 0:
                uparent = np.unravel_index(parent,np.shape(dem))
                pits[uparent] = pID 
                k = j + np.size(parent)
                if k > (chunk-1):
                    cellIndexes[p].append(np.empty((chunk,2),dtype= object))
                    chunk += 50 
                cellIndexes[p][j+1:k+1] = np.swapaxes(uparent,0,1)
                j = k  
            i = i + 1
           
        cellIndexes[p] = np.delete(cellIndexes[p],range(j+1,chunk),axis = 0) #Delete Empty or NaN array 
        cellIndexes[p] = cellIndexes[p].astype(int)
        areaCount[p] = j
    
    
    #EDGE PITCELL
    for p in range(0,np.shape(edgePitCell)[1]):
        edgeID = edgePitId[p]
        j = 0
        i = 0
        chunk = 50
        edgeIndexes = np.empty((chunk,2), dtype = object) 
        
        edgeIndexes[0][0] = edgePitCell[0][p]
        edgeIndexes[0][1] = edgePitCell[1][p]
                
        while i <= j:
            parent = flow_direction_parent[edgeIndexes[i][0]][edgeIndexes[i][1]]
            if len(parent) != 0:
                uparent = np.unravel_index(parent,np.shape(dem))
                pits[uparent] = edgeID 
                k = j + np.size(parent)
                if k > (chunk-1):
                    edgeIndexes.append(np.empty((chunk,2), dtype = object))
                    chunk += 50 
                edgeIndexes[j+1:k+1] = np.swapaxes(uparent,0,1)
                j = k  
            i = i + 1   
    
    indexes = indexes_list = list(range(0,np.size(flow_direction)))
    indexes = np.reshape(indexes_list,np.shape(flow_direction))
    pairs = [ [] for _ in range(pit_count)] #create a list within an list
    [numrows, numcols] = np.shape(dem)
    for p in range(0,np.size(cellIndexes)):
        pairs[p] = np.empty((np.shape(cellIndexes)[0]*8,4), dtype = int)
        l = 0
        for i in range(0,np.shape(cellIndexes[p])[0]): #Walk Through Indices To Check
            [r, c] = [cellIndexes[p][i][0],cellIndexes[p][i][1]]
           # print('cellIndexes[p][i]',cellIndexes[p][i])
           # print('Loop', i,'r&c',[r,c])
            for x in range(-1,2):
                for y in range(-1,2):
                    if x == 0 and y == 0:
                        continue #skip the center cell
                    if (r+y) >= (numrows-1) or (r+y)<0 or (c+x) >= (numcols-1) or (c+x)< 0:
                        continue #skip neighbors outside of the dem range
                    if np.isnan(dem[r+y][c+x]):
                        continue
                    if pits[r+y][c+x] != pits[r][c]:
                        pairs[p][l] = [r,c,r+y,c+x]
                        l = l + 1
        pairs[p] = np.delete(pairs[p],range(l,np.shape(pairs[p])[0]),axis = 0)
        
        interior_rc =pairs[p][:,0:2]
        exterior_rc = pairs[p][:,2:4]
        #print('interior_rc',interior_rc,'exterior_rc',exterior_rc)
        interiorDEM = dem[interior_rc[:,0],interior_rc[:,1]]
        exteriorDEM = dem[exterior_rc[:,0],exterior_rc[:,1]]
        #print('exteriorDEM',exteriorDEM)
        #print('interiorDEM',interiorDEM)
        
        cmax = np.empty(np.shape(interiorDEM))
        for i in range(0,len(interiorDEM)):
            cmax[i] = np.max([exteriorDEM[i],interiorDEM[i]])
        #print('MAX',cmax)
        [val,cord] = [np.min(cmax), np.argmin(cmax)] 
        #print([val,cord])
        spilloverElevation[p] = val
        cellOverflowInto[p] = exterior_rc[cord]
        
        #print(dem[cellIndexes[p].astype(int)])
        lessThans = [dem[cellIndexes[p][:,0],cellIndexes[p][:,1]] <= spilloverElevation[p]] 
        lessThans = cellIndexes[p][lessThans]
        #print('lessThans',lessThans)
        #print('DEM',dem[lessThans[:,0],lessThans[:,1]])
        volume[p] = np.sum((spilloverElevation[p] - dem[lessThans[:,0],lessThans[:,1]])*cellSize*cellSize)
        #print('loop', p, 'VOLUME',volume)
        #print('pits',pits)
        #print('Spill',spilloverElevation)
        
        filledVolume[p] = 0
        vca[p] = volume[p]/((cellSize^2)*areaCount[p])
        print(vca[p])
        
        if vca[p] < 0:
            vca[p] = np.infty
        
        
dem = np.array([[4, 7, 3, 7, 8, 8, 5, 2, 9, 8],
 [0, 8, 2, 4, 6, 4, 7, 3, 8, 5],
 [4, 5, 9, 9, 8, 3, 6, 4, 6, 6],
 [6, 1, 0, 5, 7, 7, 5, 8, 8, 7],
 [9, 9, 3, 5, 9, 7, 6, 9, 7, 2],
 [0, 2, 8, 0, 4, 0, 1, 2, 8, 7],
 [6, 9, 2, 9, 5, 5, 9, 1, 5, 9],
 [4, 3, 6, 9, 1, 3, 0, 8, 1, 5],
 [0, 1, 3, 7, 1, 9, 9, 4, 2, 8],
 [1, 4, 4, 6, 4, 4, 8, 9, 5, 9]])

flow_direction = np.array([[10., 10., 12.,  2., 13., 15.,  7., -2.,  7., 19.],
 [-2., 10., -1., 12., 25., 25., 17.,  7., 17., -2.],
 [10., 31., 32., 32., 25., -1., 25., 17., 17., 19.],
 [31., 32., -1., 32., 25., 25., 25., 27., 49., 49.],
 [50., 31., 32., 53., 53., 55., 56., 57., 49., -2.],
 [-2., 50., 53., -1., 53., -1., 55., 56., 57., 49.],
 [50., 51., 53., 53., 74., 55., 76., 76., 67., 78.],
 [80., 80., 62., 74., -1., 76., -1., 76., -1., 78.],
 [-2., 80., 81., 84., -1., 84., 76., 76., 78., 88.],
 [80., 81., 81., 84., 84., 84., 95., 87., 88., 88.]])

flow_direction_parent = np.array([[list([]), list([]), list([3]), list([]), list([]), list([]), list([]),
  list([6, 8, 17]), list([]), list([])],
 [list([0, 1, 11, 20]), list([]), list([2, 13]), list([4]), list([]),
  list([5]), list([]), list([16, 18, 27, 28]), list([]), list([9, 29])],
 [list([]), list([]), list([]), list([]), list([]),
  list([14, 15, 24, 26, 34, 35, 36]), list([]), list([37]), list([]),
  list([])],
 [list([]), list([21, 30, 41]), list([22, 23, 31, 33, 42]), list([]),
  list([]), list([]), list([]), list([]), list([]), list([])],
 [list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]),
  list([]), list([38, 39, 48, 59])],
 [list([40, 51, 60]), list([61]), list([]), list([43, 44, 52, 54, 62, 63]),
  list([]), list([45, 56, 65]), list([46, 57]), list([47, 58]), list([]),
  list([])],
 [list([]), list([]), list([72]), list([]), list([]), list([]), list([]),
  list([68]), list([]), list([])],
 [list([]), list([]), list([]), list([]), list([64, 73]), list([]),
  list([66, 67, 75, 77, 86, 87]), list([]), list([69, 79, 88]), list([])],
 [list([70, 71, 81, 90]), list([82, 91, 92]), list([]), list([]),
  list([83, 85, 93, 94, 95]), list([]), list([]), list([97]),
  list([89, 98, 99]), list([])],
 [list([]), list([]), list([]), list([]), list([]), list([96]), list([]),
  list([]), list([]), list([])]], dtype = object)

cellSize = 10
getDepression(dem, flow_direction, flow_direction_parent, cellSize)
