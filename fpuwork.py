
#to do
# - Optimize memory for SNA draw by removing the repeated links
# - Check the different coding of Arrival, departure, etc.
# - Remove/Add necessary variables
# - pickle object for future use
# - parallel process for Levenshtein

import sys, getopt, csv
import os
import pickle
#parse(commands.getoutput("date"))
from dateutil.parser import *
from collections import OrderedDict
from operator import itemgetter
from datetime import datetime
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from scipy.cluster.vq import kmeans,vq
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder

#import nltk
from Levenshtein import distance as LevDistance



import matplotlib.pyplot as plt

BASEDIR = os.path.dirname(os.path.realpath(__file__))

INPUT = BASEDIR + "\\fpuinput\\fputestdata_dedup_sub.csv"
WORKINGDIR = BASEDIR + "\\fpuworking"
OUTPUT = BASEDIR + "\\fpuoutput"





from collections import OrderedDict

class Person():

    def __init__(self):
        self.vd_id = 0
        self.person_id = []
        self.fin_id = []
        self.DOB = None
        self.age = None
        self.gender = None
        self.countryissue = []

        self.eventLines = []

        self.eventHistory = []

        self.suspiciousRecord = []

        self.suspicious = False

        self.verifiedfraud = False

        self.vector = {}

        self.dedupaddresslist = []

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
            self.DOB = parse(lineobj['DOB'].replace("-", "-1-"))
        except:
            self.DOB = None
        if not self.DOB:
            try:
                self.DOB = parse(lineobj['DOB'].split("-")[1] + "/1/" + lineobj['DOB'].split("-")[0])
            except:
                self.DOB = None
                print "failed to parse"
        try:
            if self.DOB:
                self.age = int((datetime.now() - self.DOB).days/365)
            else:
                self.age = None
        except:
            print "failed to calulcate age"
            self.age = None


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
                        lookback = Event(tempevent, self)
                        self.eventHistory.append(lookback)
                    elif (tempevent['EVENT_TYPE'] == "Departure"):
                        lookback.appendDeparture(tempevent)
                    else:
                        #create Event and save to lookback
                        lookback = Event(tempevent, self)
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
                histevent.suspicious = True
                self.suspiciousRecord.append("Has arrival but not departure " +  str(histevent.arrivalID))

            if not histevent.arrivalID and histevent.departureID:
                self.suspicious = True
                histevent.suspicious = True
                self.suspiciousRecord.append("Has depature but not arrival " +  str(histevent.departureID))

            if histevent.arrivalDate and histevent.departureDate:
                numdays = histevent.calculateLengthofStay()
                if numdays > 180:
                    self.suspicious = True
                    histevent.suspicious = True
                    self.suspiciousRecord.append("Has stayed " + str(numdays) + " days for" +  str(histevent.arrivalID) + " " + str(histevent.departureID))

            if (histevent.departureOverstay not in  ("null",None) and int(histevent.departureOverstay) > 1) or (histevent.arrivalOverstay not in ("null",None) and int(histevent.arrivalOverstay) > 1):
                self.suspicious = True
                self.verifiedfraud = True
                histevent.suspicious = True
                histevent.verifiedfraud = True
                self.suspiciousRecord.append("Has field of overstay " +  str(histevent.arrivalID) + " " +  str(histevent.departureID))

    def getEvents(self):
        return self.eventHistory

    def pushVector(self,index,value):
        self.vector[index] = value

    def vectorize(self):
        self.vector['numtrips'] = len(self.eventHistory)
        lenofstay = []
        for event in self.eventHistory:
            if event.lengthofstay != None:
                lenofstay.append(event.lengthofstay)
        self.vector['meanlengthofstay'] = np.mean(np.trim_zeros(np.array(lenofstay)))
        self.vector['age'] = self.age
        self.vector['travelevents'] = 0
        for event in self.eventHistory:
            if (event.suspicious):
                self.vector['suspiciousevent'] +=1

    def getVectorDict(self):
        return self.vector

    def runDeDupAddress(self):
        baseAddresses = []
        for hist in self.eventHistory:
            baseAddresses.append(hist.destAddress + ", " + hist.destCity + ", " + hist.destState)

        self.dedupaddresslist = []
        for baseAddressIndex in range(len(baseAddresses)):
            keep = True
            for checkAddressIndex in range(len(baseAddresses)):
                if checkAddressIndex == baseAddressIndex:
                    continue
                        #number of chars that ened to be edited in order for it to be the same string
                if LevDistance(baseAddresses[baseAddressIndex], baseAddresses[checkAddressIndex]) < 5:
                    keep = False
                    break
            if keep:
                self.dedupaddresslist.append(baseAddresses[baseAddressIndex])


    def getMatchingAddress(self, personList):

        matchlist = []
        for person in personList:
            if person.vd_id == self.vd_id:
                continue

            for targetaddress in person.dedupaddresslist:
                for baseaddress in self.dedupaddresslist:
                    if LevDistance(baseaddress, targetaddress) < 5:
                        if person.vd_id not in matchlist:
                            matchlist.append(person.vd_id)

        return matchlist










#an event is an arrival and depart date together
class Event():

    def __init__(self, lineobj, person):
        self.person = person
        self.docCountry = lineobj['DOCUMENT_COUNTRY_ISSUE']
        self.docType = lineobj['DOCUMENT_TYPE']
        self.LocationCode = lineobj['LOCATION_CODE']
        self.admissionClass = lineobj['ADMISSION_CLASS']
        self.auditDate = lineobj['AUD']
        self.carrierCode = lineobj['CARRIER_CODE']
        self.flightNum = lineobj['FLIGHT_NUMBER']
        self.psgStatus = lineobj['PSNGR_STATUS']
        self.suspicious = False
        self.verifiedfraud = False
        self.vector = {}
        if lineobj['EVENT_TYPE'] == "Departure":
            self.departureDate = parse(lineobj['EVENT_DATE'])
            self.departureOverstay = lineobj['OVER_STAY_DAYS']
            self.departureID = lineobj['EVENT_ID']
            try:
                something = self.arrivalID
            except:
                self.arrivalDate = None
                self.arrivalOverstay = None
                self.arrivalID = None
        else:
            self.arrivalDate = parse(lineobj['EVENT_DATE'])
            self.arrivalOverstay = lineobj['OVER_STAY_DAYS']
            self.arrivalID = lineobj['EVENT_ID']
            self.departureDate = None
            self.departureOverstay = None
            self.departureID = None
        self.claimForm = lineobj['CLAIMFORMTYPE']
        if (lineobj['I94_NUMBER'] not in  ("null",0)):
            self.i94Num = True
        else:
            self.i94Num = False
        self.reconCode = lineobj['RECONCILIATION_CODE']
        self.actionCode = lineobj['PRIMARY_ACTION_CODE']
        self.destAddress = lineobj['DEST_ADDRESS']
        self.destCity = lineobj['DEST_CITY_NAME']
        self.destState = lineobj['DEST_STATE_CODE']
        self.departAirport = lineobj['DEPARTURE_AIRPORT']
        self.travelMode = lineobj['TRAVEL_MODE']
        self.lengthofstay = None

    def appendDeparture(self, lineobj):
        self.departureDate = parse(lineobj['EVENT_DATE'])
        self.departureOverstay = lineobj['OVER_STAY_DAYS']
        self.departureID = lineobj['EVENT_ID']
        return self

    def calculateLengthofStay(self):
        self.lengthofstay = int((self.departureDate - self.arrivalDate).days)
        return self.lengthofstay

    def pushVector(self,index,value):
        self.vector[index] = value

    def vectorize(self):
        self.vector['i94Num'] = self.i94Num
        self.vector['lengthofstay'] = self.lengthofstay

    def getVectorDict(self):
        return self.vector


masterEventsList  = []
def getAllEvents(masterEventsList):
    for person in personList:
        masterEventsList += person.getEvents()
    return masterEventsList



###suspicious travel vs suspicious people


genderEncoder = LabelEncoder()

eventEncoders = {"docCountry": LabelEncoder(),\
                 "docType": LabelEncoder(),\
                 "LocationCode": LabelEncoder(),\
                 "admissionClass": LabelEncoder(),\
                 "carrierCode": LabelEncoder(),\
                 "psgStatus": LabelEncoder(),\
                 "destAddress": LabelEncoder(),\
                 "destCity": LabelEncoder(),\
                 "destState": LabelEncoder()
                 }

def buildEncodedLabels(masterEventsList):

    #gender
    temppersonobj = []
    for person in personList:
        temppersonobj.append(person.gender)

    genderTranformed = genderEncoder.fit_transform(temppersonobj)

    masterEventsList = getAllEvents(masterEventsList)

    for en in eventEncoders.keys():
        templist = []
        for tempevent in masterEventsList:
            templist.append(getattr(tempevent,en))
        eventEncoders[en].fit(templist)

    return (genderTranformed, eventEncoders)




#can now
# >>> le.transform(["tokyo", "tokyo", "paris"])
# array([2, 2, 1])
# >>> list(le.inverse_transform([2, 2, 1]))



def vectorizeEvents(masterEventsList):
    for event in masterEventsList:
        event.vectorize()

    for en in eventEncoders.keys():
        #get all event attributes
        temparray = []
        for event in masterEventsList:
            temparray.append(getattr(event, en))
        resultsset = eventEncoders[en].transform(temparray)
        for theresult, event in zip(resultsset, masterEventsList):
            event.pushVector(en, theresult)





def vectorizePeople(genderTranformed):
    for person, gendervalue in zip(personList, genderTranformed):
        person.vectorize()
        person.pushVector("gender", gendervalue)


def printSuspiciousNarrative(suspiciousnarrative):
    with open(OUTPUT + "\\suspiciousnarrative.txt", 'wb') as f:
        for index in suspiciousnarrative.keys():
            f.write(index + "\n")
            for event in suspiciousnarrative[index]:
                f.write(event +"\n")

def printSummaryOutput(maincounts):
    print OUTPUT + "\\summaryoutput.txt"
    with open(OUTPUT + "\\summaryoutput.txt", 'wb') as f:
        for index in maincounts.keys():
            f.write(index + "  ***************\n")
            f.write("Total Count: " + str(maincounts[index]['totnum']) + '\n')
            f.write("Total Females: " + str(maincounts[index]['numfemales']) + '\n')
            f.write("Total Males: " + str(maincounts[index]['nummales']) + '\n')

            f.write("Mean Length of Stay: " + str(np.mean(np.trim_zeros(np.nan_to_num(maincounts[index]['meanlengthofstay'])))) + '\n')
            f.write("Mean Age: " + str(np.mean(np.trim_zeros(maincounts[index]['meanage']))) + '\n')
            f.write("Mean Travel Events: " + str(np.mean(np.trim_zeros(maincounts[index]['meantravelevents']))) + '\n')



def summarystatsoutput(masterEventsList):
    print "calculating summarystats"
    suspiciousnarrative = {}
    maincounts = {"people":{"totnum": 0, "numfemales":0, "nummales":0,"meanlengthofstay":[], "meanage":[], "meantravelevents":[]},\
                  "suspicious":{"totnum": 0, "numfemales":0, "nummales":0,"meanlengthofstay":[], "meanage":[], "meantravelevents":[]},\
                  "fraud":{"totnum": 0, "numfemales":0, "nummales":0,"meanlengthofstay":[], "meanage":[], "meantravelevents":[]}}
    for person in personList:
        if person.verifiedfraud:
            maincounts['fraud']['totnum'] +=1
            if person.gender == "M":
                maincounts['fraud']['nummales'] +=1
            else:
                maincounts['fraud']['numfemales'] +=1
            maincounts['fraud']['meanlengthofstay'].append(person.vector['meanlengthofstay'])
            maincounts['fraud']['meanage'].append(person.vector['age'])
            maincounts['fraud']['meantravelevents'].append(len(person.eventHistory))

        if person.suspicious:
            maincounts['suspicious']['totnum'] +=1
            if person.gender == "M":
                maincounts['suspicious']['nummales'] +=1
            else:
                maincounts['suspicious']['numfemales'] +=1

            maincounts['suspicious']['meanlengthofstay'].append(person.vector['meanlengthofstay'])
            maincounts['suspicious']['meanage'].append(person.vector['age'])
            maincounts['suspicious']['meantravelevents'].append(len(person.eventHistory))

            #collect the narrative
            suspiciousnarrative[person.vd_id] = person.suspiciousRecord

        maincounts['people']['totnum'] +=1
        if person.gender == "M":
            maincounts['people']['nummales'] +=1
        else:
            maincounts['people']['numfemales'] +=1

        maincounts['people']['meanlengthofstay'].append(person.vector['meanlengthofstay'])
        maincounts['people']['meanage'].append(person.vector['age'])
        maincounts['people']['meantravelevents'].append(len(person.eventHistory))

    printSuspiciousNarrative(suspiciousnarrative)

    printSummaryOutput(maincounts)






    #print narrative





#start on the stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from matplotlib.mlab import PCA


def getColors(thelist):
    colors=[]
    for item in thelist:
        if item.verifiedfraud:
            colors.append('r')
        elif item.suspicious:
            colors.append('b')
        else:
            colors.append('g')

    return colors

def PCAevents(eventDataVectorized):
    PCAcalculator  = PCA( n_components=2)
    fitModel = PCAcalculator.fit_transform(eventDataVectorized)
    print "fit model", fitModel
    print len(fitModel)
    print "my x", len(fitModel[:,0])
    print "my y", len(fitModel[:,1])
    colors = getColors(masterEventsList)

    plt.scatter(fitModel[:, 0], fitModel[:, 1], alpha=.5, s=20, color=colors)
    plt.show()


def PCApeople(peopleDataVectorized):
    PCAcalculator  = PCA( n_components=2)
    fitModel = PCAcalculator.fit_transform(peopleDataVectorized)
    print "fit model", fitModel
    print len(fitModel)
    print "my x", len(fitModel[:,0])
    print "my y", len(fitModel[:,1])
    colors = getColors(personList)

    plt.scatter(fitModel[:, 0], fitModel[:, 1], alpha=.5, s=20, color=colors)
    plt.show()

import networkx as nx

def performSNA(personList):
    #find common addresses
    addressG = nx.Graph()
    traveldates = nx.Graph()
    print "deduping addresses"
    count = 0
    for person in personList:
        print float(count)/float(len(personList))*100
        person.runDeDupAddress()
        addressG.add_node(str(person.vd_id))
        traveldates.add_node(str(person.vd_id))

    print "findnig matches"
    count = 0
    for person in personList:
        print float(count)/float(len(personList))*100
        matchinglist = person.getMatchingAddress(personList)
        for match in matchinglist:
            addressG.add_edge(person.vd_id, match)
        count += 1
    plt.figure(1)
    colors = getColors(personList)
    nx.draw_spring(addressG, node_size =20, node_color = colors, linewidths =.1, alpha =.5, edge_color ="grey")
    plt.show()














# >>> from sklearn.datasets import load_iris
# >>> from sklearn.feature_selection import SelectKBest
# >>> from sklearn.feature_selection import chi2
# >>> iris = load_iris()
# >>> X, y = iris.data, iris.target
# >>> X.shape
# (150, 4)
# >>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# >>> X_new.shape
# (150, 2)

#sklearn.feature_selection.univariate_selection.f_regression










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


genderTranformed, eventEncoders = buildEncodedLabels(masterEventsList)

vectorizeEvents(masterEventsList)

vectorizePeople(genderTranformed)

summarystatsoutput(masterEventsList)




eventDictVector = []
for events in masterEventsList:
    eventDictVector.append(events.getVectorDict())

eventVectorizer = DictVectorizer(sparse=False)

eventDataVectorized = np.nan_to_num(eventVectorizer.fit_transform(eventDictVector))

PCAevents(eventDataVectorized)



#save on memory
del eventDictVector
del eventDataVectorized


peopleDictVector = []
for person in personList:
    peopleDictVector.append(person.getVectorDict())

peopleVectorizer = DictVectorizer(sparse=False)

peopleDataVectorized = np.nan_to_num(peopleVectorizer.fit_transform(peopleDictVector))

PCAevents(peopleDataVectorized)


del peopleDictVector
del peopleVectorizer




#check variables

#apply model

#common address listings
performSNA(personList)



#SNA


sys.exit(1)
