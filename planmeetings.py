

import sys, getopt, csv
import os
import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from scipy.cluster.vq import kmeans,vq
from scipy.spatial import distance

BASEDIR = os.path.dirname(os.path.realpath(__file__))
print BASEDIR
INPUT = BASEDIR + "\\input\\Pre-Orlando_Matching_Data_Cleaned.csv"
WORKINGDIR = BASEDIR + "\\working"
OUTPUT = BASEDIR + "\\output"

#higher the weight the less it will impact the matching
#scaled to 15
SECTOR_WEIGHT = 10

#scaled to 15
ACTIVITY_WEIGHT = 10

#scaled at 7
CLIENTS_WEIGHT = 20





def vectorizeColumn(inputobj, fieldname):
    dataset = []
    #this could be more efficient
    for feature in inputobj:
        dataset.append(feature[fieldname])



    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    km = KMeans(n_clusters=15, init='k-means++', max_iter=100, n_init=1,verbose=1)



    X_test = vectorizer.fit_transform(dataset)
    km.fit(X_test)
    return km.labels_


def normClients(featurevalue):
    if featurevalue == '':
        return 1
    featurevalue = int(featurevalue.replace(",",""))
    print featurevalue
    if featurevalue < 100:
        return 1
    elif featurevalue < 500:
        return 2
    if featurevalue < 1000:
        return 3
    elif featurevalue < 2500:
        return 4
    if featurevalue < 5000:
        return 5
    elif featurevalue < 15000:
        return 6
    else:
        return 7



def vectorizeData(inputobj):


    calcset = []
    counter = 0
    for feature in inputobj:
        extras = {"Location":feature['Location'], "Eng":feature['Eng.'],"Spa":feature['Spa.']}
        calcset.append({"label":feature['Institution'], "data":{}, "extras":extras})
        inputobj[counter]['bow_sector'] = " ".join([feature['Sector 1'],feature['Sector 2'],feature['Sector 3'],feature['Sector 4']])
        inputobj[counter]['bow_activity'] = " ".join([feature['Activity 1'],feature['Activity 2'],feature['Activity 3'],feature['Activity 4']])
        inputobj[counter]['client_calc'] = normClients(feature[' Clients '])
        counter += 1

    vecresult_sector = vectorizeColumn(inputobj, "bow_sector")
    vecresult_activity = vectorizeColumn(inputobj, "bow_sector")
    for result_sector, result_activity, featureorig,calcset_index  in zip(vecresult_sector, vecresult_activity, inputobj, range(len(calcset))):
        datadict = {"sector":result_sector, "activity":result_activity, "clients": featureorig['client_calc']}
        calcset[calcset_index]['data'] = datadict

    return calcset



def calculateMeetings(calcset):

    v = DictVectorizer(sparse=False)
    datasetdict = []
    for feature in calcset:
        datasetdict.append(feature['data'])

    X = v.fit_transform(datasetdict)
    print "There will be ", int(len(datasetdict)/4), "groups"
    km = KMeans(n_clusters=int(len(datasetdict)/4), init='k-means++', max_iter=100, n_init=1,verbose=1)
    km.fit(X)

    centroids,_ = kmeans(X,int(len(datasetdict)/4))
    # assign each sample to a cluster
    idx,_ = vq(X,centroids)

    groupingresult = {}
    for index, grouping in zip(range(len(calcset)),idx):
        if str(grouping) in groupingresult.keys():
            groupingresult[str(grouping)].append(calcset[index]['label'])
        else:
            groupingresult[str(grouping)] = [calcset[index]['label']]

    return groupingresult
    #get the distances between the centroids

    #get distance between the sets
    #dst = distance.euclidean(a,b)




    #constraints





    #fit to k-means++ with n/k components
    #calculate distance to centroid and the preferred next centroids
    #swap to fill
    #iterate for each to return variance
        #select candidate by distance to their centroid
        #calculate distance to points that are close
        #swap candidates and calculate new variance
            #if better then keep it
            #recalculate the distance to centroid and preferred next centroids
    return

def getInputObj():
    with open (INPUT, 'rb') as csvfile:
        thereader = csv.reader(csvfile)
        isfirst = True
        header = []
        outputobj = []
        for row in thereader:
            if isfirst:
                header = row
                isfirst = False
                continue
            outputobj.append(dict(zip(header, row)))
    return outputobj



def main(argv):

    try:
        opts, args = getopt.getopt(argv,"c:",["command="])
    except getopt.GetoptError:
        print 'planmeetings.py -c <command> (cleandata, calculate)'
        sys.exit(2)
    for opt, arg in opts:
      if opt == '-c':
          if arg == "vectorize":
              inputobj = getInputObj()
              calcset = vectorizeData(inputobj)
              with open(WORKINGDIR + "\\working.pickle", 'wb') as pickfile:
                  pickle.dump( calcset, pickfile)
              print "success"
          elif arg == "calculate":
              try:
                  with open(WORKINGDIR + "\\working.pickle", 'rb') as pickfile:
                      calcset = pickle.load(pickfile)
              except:
                  print "could not find worknig file, try vectorizing first"
                  sys.exit()
              groupingresult = calculateMeetings(calcset)
              with open(OUTPUT + "\\output.txt", 'wb') as outputfile:
                  for group_index in groupingresult.keys():
                      outputfile.write(group_index + "\n")
                      outputfile.write("-----------------" + "\n")
                      for theresult in groupingresult[group_index]:
                          outputfile.write(theresult + "\n")


          else:
              print 'planmeetings.py -c <command> (cleandata, calculate)'
              sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])


