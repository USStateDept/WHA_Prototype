

import sys, getopt, csv
import os
import pickle
#parse(commands.getoutput("date"))
from dateutil.parser import *
from collections import OrderedDict
from operator import itemgetter
from datetime import datetime


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from scipy.cluster.vq import kmeans,vq
from scipy.spatial import distance

BASEDIR = os.path.dirname(os.path.realpath(__file__))

INPUT = BASEDIR + "\\fpuinput\\fputestdata_dedup_sub.csv"
WORKINGDIR = BASEDIR + "\\working"
OUTPUT = BASEDIR + "\\output"





from collections import OrderedDict

class Person():

    def __init__(self):
        self.vd_id = 0
        self.person_id = []
        self.fin_id = []
        self.DOB = None
        self.gender = None
        self.countryissue = []

        self.eventLines = []

        self.eventHistory = []

        self.suspiciousRecord = []

        self.suspicious = False

        self.verifiedfraud = False

    def get_or_create(self, lineobj):
        for person in personList:
            if lineobj['ID'] == person.vd_id:
                if (lineobj['PERSON_ID_FK'] not in person.person_id and lineobj['PERSON_ID_FK'] != "null"):
                    person.person_id.append(lineobj['PERSON_ID_FK'])
                if (lineobj['FIN'] not in person.fin_id and lineobj['FIN'] != "null"):
                    person.fin_id.append(lineobj['FIN'])
                if (lineobj['DOCUMENT_COUNTRY_ISSUE'] not in person.countryissue):
                    person.countryissue.append(lineobj['DOCUMENT_COUNTRY_ISSUE'])
                return person



        self.vd_id = lineobj['ID']
        if lineobj['PERSON_ID_FK'] != "null":
            self.person_id.append(lineobj['PERSON_ID_FK'])
        if lineobj['FIN'] != "null":
            self.fin_id.append(lineobj['FIN'])

        try:
            self.DOB = parse(lineobj['DOB'].replace("-", "/"))
        except:
            print lineobj['DOB'].replace("-", "/")
            print "failed DOB parse"
            self.DOB = None
        self.countryissue.append(lineobj['DOCUMENT_COUNTRY_ISSUE'])
        self.gender = lineobj['GENDER_CODE']
        personList.append(self)
        return self


    def addEventLine(self, lineobj):
        self.eventLines.append(lineobj)
        return self

    def createHistory(self):
        #id date
        tempeventIDs = {}
        for tempEvent in self.eventLines:
            #get seconds so we can sort
            eventDate = parse(tempEvent['EVENT_DATE'])
            eventDateSeconds = eventDate - datetime(1970,1,1)
            tempeventIDs[tempEvent['EVENT_ID']] = int(eventDateSeconds.total_seconds())


        orderlistd = OrderedDict(sorted(tempeventIDs.items(), key=itemgetter(1)))
        lookback = None
        for eventdatetime in orderlistd:
            for tempevent in self.eventLines:
                if (tempevent['EVENT_ID'] == eventdatetime):
                    #now that we have the object
                    if not lookback:
                        #create a new event
                        lookback = Event(tempevent)
                        self.eventHistory.append(lookback)
                    elif (tempevent['EVENT_TYPE'] == "Departure"):
                        lookback.appendDeparture(tempevent)
                    else:
                        #create Event and save to lookback
                        lookback = Event(tempevent)
                        self.eventHistory.append(lookback)


        #take ll eventLines and put them together to have an eventHistory
        return self

    def buildFINCheck(self):
        if (len(self.person_id) >1):
            self.suspicious = True
            self.suspiciousRecord.append("Has multiple person_ids" +  ",".join(self.person_id))
        if (len(self.fin_id) > 1):
            self.suspicious = True
            self.suspiciousRecord.append("Has multiple find_ids" +  ",".join(self.fin_id))


    def buildOverstayCheck(self):
        #arrival without departure
        for histevent in self.eventHistory:
            if histevent.arrivalID and not histevent.departureID:
                self.suspicious = True
                self.suspiciousRecord.append("Has arrival but not departure " +  str(histevent.arrivalID))

            if not histevent.arrivalID and histevent.departureID:
                self.suspicious = True
                self.suspiciousRecord.append("Has depature but not arrival " +  str(histevent.departureID))

            if histevent.arrivalDate and histevent.departureDate:
                if int((histevent.departureDate - histevent.arrivalDate).days) > 180:
                    self.suspicious = True
                    self.suspiciousRecord.append("Has stayed longer than 180 days for" +  str(histevent.arrivalID) + " " + str(histevent.departureID))

            if (histevent.departureOverstay != "null" and int(histevent.departureOverstay) > 1) or (histevent.arrivalOverstay != "null" and int(histevent.arrivalOverstay) > 1):
                self.suspicious = True
                self.verifiedfraud = True
                self.suspiciousRecord.append("Has field of overstay " +  str(histevent.arrivalID) + " " +  str(histevent.departureID))










#an event is an arrival and depart date together
class Event():

    def __init__(self, lineobj):
        self.docCountry = lineobj['DOCUMENT_COUNTRY_ISSUE']
        self.docType = lineobj['DOCUMENT_TYPE']
        self.LocationCode = lineobj['LOCATION_CODE']
        self.admissionClass = lineobj['ADMISSION_CLASS']
        self.auditDate = lineobj['AUD']
        self.carrierCode = lineobj['CARRIER_CODE']
        self.flightNum = lineobj['FLIGHT_NUMBER']
        self.psgStatus = lineobj['PSNGR_STATUS']
        if lineobj['EVENT_TYPE'] == "Departure":
            self.departureDate = parse(lineobj['EVENT_DATE'])
            self.departureOverstay = lineobj['OVER_STAY_DAYS']
            self.departureID = lineobj['EVENT_ID']
        else:
            self.arrivalDate = parse(lineobj['EVENT_DATE'])
            self.arrivalOverstay = lineobj['OVER_STAY_DAYS']
            self.arrivalID = lineobj['EVENT_ID']
        self.claimForm = lineobj['CLAIMFORMTYPE']
        self.i94Num = lineobj['I94_NUMBER']
        self.reconCode = lineobj['RECONCILIATION_CODE']
        self.actionCode = lineobj['PRIMARY_ACTION_CODE']
        self.destAddress = lineobj['DEST_ADDRESS']
        self.destCity = lineobj['DEST_CITY_NAME']
        self.destState = lineobj['DEST_STATE_CODE']
        self.departAirport = lineobj['DEPARTURE_AIRPORT']
        self.travelMode = lineobj['TRAVEL_MODE']

    def appendDeparture(self, lineobj):
        self.departureDate = parse(lineobj['EVENT_DATE'])
        self.departureOverstay = lineobj['OVER_STAY_DAYS']
        self.departureID = lineobj['EVENT_ID']
        return self



personList = []

#create the base stuctures
with open(INPUT, 'rb') as csvfile:
    thereader = csv.reader(csvfile)
    isfirst = True
    header = []
    outputobj = []
    for row in thereader:
        if isfirst:
            header = row
            isfirst = False
            continue
        tempobj = dict(zip(header, row))


        tempPerson = Person()
        returnpersonobj = tempPerson.get_or_create(tempobj)


        returnpersonobj.addEventLine(tempobj)
#build the history
for person in personList:
    person.createHistory()


#let's look fpr suspicious folks

    #more FINs or Person IDs for person
for person in personList:
    person.buildFINCheck()

    person.buildOverstayCheck()

    #otyher checks





for person in personList:
    print person.vd_id
    print len(person.eventHistory)





for line in ADFile:
    numTot += 1
    line_lower = line.strip().lower()
    col = line.strip().split('\t')
    if line_lower in seen:
        numDup += 1
    else:
        seen.add(line_lower)
    eventList.insert(len(eventList),col[21])
    # DoB check since it's only month and year
    dob = col[2].split('/')
    if len(dob) == 2:
        dobyear = dob[1]
        dobmonth = dob[0]
    # check ID & Person_ID_FK & FIN (null in Depart)
    Person(col[0],col[1])
    personList.add(Person)
#DeDup the event List as IDs
uniqueEvents = OrderedDict.fromkeys(eventList)
print "Unique Events: ", len(uniqueEvents)
print "Duplicates: ", numDup
print "Total records read: ", numTot







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


