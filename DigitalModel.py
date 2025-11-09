import logging
import numpy
import simpy
import time
from random import Random

from supervisor_class.simulator_class.manpy.simulation.Globals import G
from supervisor_class.simulator_class.manpy.simulation.Order import Order
import supervisor_class.simulator_class.manpy.simulation.PrintRoute as PrintRoute
import supervisor_class.simulator_class.manpy.simulation.ExcelHandler as ExcelHandler
from supervisor_class.simulator_class.manpy.simulation.ProcessingTimeList import ProcessingTimeList
from supervisor_class.simulator_class.manpy.simulation.RandomNumberGenerator import RandomNumberGenerator
import supervisor_class.simulator_class.manpy.simulation.Globals as Globals

logger = logging.getLogger("manpy.platform")
numpy.seterr(all="raise")

class DigitalModel:
    def __init__(self, execModel, ID):
        # The graphModel must be given as a dict
        if not isinstance(execModel, dict):
            raise ValueError("The executable model must be given as a dict")

        # Save the dict with key 'general'
        self.general = execModel["general"]

        # Save the dict of the nodes and edges
        self.nodes = execModel["graph"]["node"]
        self.edges = execModel["graph"]["edge"]
        self.inputWIP = execModel.get("input", {})
        self.bom = self.inputWIP.get("BOM", None)

        # Save the ID of the current executable model
        self.ID = ID

        # Initialize the attribute that saves the number of WIP in a closed system
        self.numOfWIP = 0

        # Initialize the lists with all the elements of the model
        self.ObjectList = []
        self.RouterListC = []
        self.EntityListC = []

        # Initialize the model
        G.ObjList = []  # first we initialize the global lists as empty
        G.RouterList = []
        # We initialize the lists and assign them as attributes of the object
        self.createObjectResourcesAndCoreObjects()
        self.createObjectInterruptions()
        self.setTopology()
        G.ObjList = []
        G.RouterList = []
        self.Thisenv = simpy.Environment()

    def __repr__(self):
        sentence = f"Model composed of those {len(self.ObjectList)} objects \n"
        for i, a in enumerate(self.ObjectList):
            sentence += f'{a.id!r}: predecessorList = {a.previousIds}, successorList = {a.nextIds}\n'
        return sentence

    def readGeneralInput(self):
        G.numberOfReplications = int(
            self.general.get("numberOfReplications", "1")
        )  # read the number of replications / default 1
        G.maxSimTime = float(
            self.general.get("maxSimTime", "100")
        )  # get the maxSimTime / default 100
        G.trace = self.general.get(
            "trace", "No"
        )  # get trace in order to check if trace is requested
        G.console = self.general.get(
            "console", "No"
        )  # get console flag in order to check if console print is requested
        G.confidenceLevel = float(
            self.general.get("confidenceLevel", "0.95")
        )  # get the confidence level
        G.seed = self.general.get("seed")  # the seed for random number generation
        G.extraPropertyDict = self.general.get(
            "extraPropertyDict", {}
        )  # a dict to put extra properties that are
        # generic for the model
        G.initializingFlag = self.general.get(
            "initializingFlag", False
        )
        G.initializingFilename = self.general.get(
            "initializingFilename", ""
        )

    def getSuccessorList(
        self, node_id, predicate=lambda source, destination, edge_class, edge_data: True
    ):
        successor_list = []  # dummy variable that holds the list to be returned

        for edge in list(self.edges.values()):
            source = edge["source"]
            destination = edge["destination"]
            edge_class = edge["_class"]
            edge_data = edge.get("data", {})
            if source == node_id:  # for the node_id argument
                if predicate(
                    source, destination, edge_class, edge_data
                ):  # find its 'destinations' and
                    successor_list.append(
                        destination
                    )  # append it to the successor list

        # XXX We should probably not need to sort, but there is a bug that
        # prevents Topology10 to work if this sort is not used.
        successor_list.sort()
        return successor_list

    def createObjectResourcesAndCoreObjects(self):
        """
        define the lists of each object type
        """
        G.SourceList = []
        G.MachineList = []
        G.ExitList = []
        G.QueueList = []
        G.RepairmanList = []
        G.AssemblyList = []
        G.DismantleList = []
        G.ConveyorList = []
        G.MachineJobShopList = []
        G.QueueJobShopList = []
        G.ExitJobShopList = []
        G.BatchDecompositionList = []
        G.BatchSourceList = []
        G.BatchReassemblyList = []
        G.RoutingQueueList = []
        G.LineClearanceList = []
        G.EventGeneratorList = []
        G.OperatorsList = []
        G.OperatorManagedJobsList = []
        G.OperatorPoolsList = []
        G.BrokersList = []
        G.OperatedMachineList = []
        G.BatchScrapMachineList = []
        G.OrderDecompositionList = []
        G.ConditionalBufferList = []
        G.MouldAssemblyBufferList = []
        G.MouldAssemblyList = []
        G.MachineManagedJobList = []
        G.QueueManagedJobList = []
        G.ObjectResourceList = []
        G.CapacityStationBufferList = []
        G.AllocationManagementList = []
        G.CapacityStationList = []
        G.CapacityStationExitList = []
        G.CapacityStationControllerList = []

        """
        loop through all the model resources
        search for repairmen and operators in order to create them
        read the data and create them
        """

        for (element_id, element) in list(
            self.nodes.items()
        ):  # use an iterator to go through all the nodes
            element["id"] = element_id  # create a new entry for the element (dictionary)
            element = element.copy()
            for k in ("element_id", "top", "left"):
                element.pop(k, None)
                # with key 'id' and value the the element_id
            resourceClass = element["_class"]

            objectType = Globals.getClassFromName(resourceClass)
            from supervisor_class.simulator_class.manpy.simulation.ObjectResource import (
                ObjectResource,
            )  # operator pools to be created later since they use operators

            # ToDo maybe it is semantically different object
            if (
                issubclass(objectType, ObjectResource)
                and not resourceClass == "manpy.OperatorPool"
            ):
                inputDict = dict(element)
                # create the CoreObject
                objectResource = objectType(**inputDict)
                # if there already coreObjectsIds defined then append the successors to them
                if objectResource.coreObjectIds:
                    for element in self.getSuccessorList(element["id"]):
                        if not element in objectResource.coreObjectIds:
                            objectResource.coreObjectIds.append(element)
                else:
                    objectResource.coreObjectIds = self.getSuccessorList(element["id"])

        """
        loop through all the model resources
        search for operatorPools in order to create them
        read the data and create them
        """
        from supervisor_class.simulator_class.manpy.simulation.OperatorPool import OperatorPool

        for (element_id, element) in list(
            self.nodes.items()
        ):  # use an iterator to go through all the nodes
            # the key is the element_id and the second is the
            # element itself
            element = element.copy()
            element["id"] = element_id  # create a new entry for the element (dictionary)
            for k in ("element_id", "top", "left"):
                element.pop(k, None)
                # with key 'id' and value the the element_id
            resourceClass = element["_class"]
            if resourceClass == "manpy.OperatorPool":
                id = element.get(
                    "id", "not found"
                )  # get the id of the element   / default 'not_found'
                name = element.get(
                    "name", "not found"
                )  # get the name of the element / default 'not_found'
                capacity = int(element.get("capacity") or 1)
                operatorsList = []
                for (
                    operator
                ) in G.OperatorsList:  # find the operators assigned to the operatorPool
                    if id in operator.coreObjectIds:
                        operatorsList.append(operator)
                if (
                    len(operatorsList) == 0
                ):  # if the operatorsList is empty then assign no operators
                    OP = OperatorPool(
                        element_id, name, capacity
                    )  # create a operatorPool object
                else:
                    OP = OperatorPool(
                        element_id, name, capacity, operatorsList
                    )  # create a operatorPool object
                OP.coreObjectIds = self.getSuccessorList(
                    id
                )  # update the list of objects that the operators of the operatorPool operate
                for operator in operatorsList:
                    operator.coreObjectIds = (
                        OP.coreObjectIds
                    )  # update the list of objects that the operators operate
                G.OperatorPoolsList.append(OP)  # add the operatorPool to the RepairmanList
        """
        loop through all the elements
        read the data and create them
        """
        for (element_id, element) in list(self.nodes.items()):
            element = element.copy()
            element["id"] = element_id
            element.setdefault("name", element_id)

            for k in ("element_id", "top", "left"):
                element.pop(k, None)
            objClass = element["_class"]
            objectType = Globals.getClassFromName(objClass)

            from supervisor_class.simulator_class.manpy.simulation.CoreObject import CoreObject

            if issubclass(objectType, CoreObject):
                inputDict = dict(element)
                coreObject = objectType(**inputDict)
                self.ObjectList.append(coreObject)
                coreObject.nextIds = self.getSuccessorList(element["id"])
                coreObject.nextPartIds = self.getSuccessorList(
                    element["id"],
                    lambda source, destination, edge_class, edge_data: edge_data.get(
                        "entity", {}
                    )
                    == "Part",
                )
                coreObject.nextFrameIds = self.getSuccessorList(
                    element["id"],
                    lambda source, destination, edge_class, edge_data: edge_data.get(
                        "entity", {}
                    )
                    == "Frame",
                )

        for element in self.ObjectList:
            for nextId in element.nextIds:
                for possible_successor in self.ObjectList:
                    if possible_successor.id == nextId:
                        possible_successor.previousIds.append(element.id)

    def createObjectInterruptions(self):
        G.ObjectInterruptionList = []
        G.ScheduledMaintenanceList = []
        G.FailureList = []
        G.BreakList = []
        G.ShiftSchedulerList = []
        G.ScheduledBreakList = []
        G.EventGeneratorList = []
        G.CapacityStationControllerList = []
        G.PeriodicMaintenanceList = []

        for (element_id, element) in list(
            self.nodes.items()
        ):  # use an iterator to go through all the nodes
            element["id"] = element_id  # create a new entry for the element (dictionary)
            objClass = element.get(
                "_class", "not found"
            )  # get the class type of the element
            from supervisor_class.simulator_class.manpy.simulation.ObjectInterruption import ObjectInterruption

            objClass = element["_class"]
            objectType = Globals.getClassFromName(objClass)

            if issubclass(objectType, ObjectInterruption):  # check the object type
                inputDict = dict(element)
                objectInterruption = objectType(**inputDict)
                if not "OperatorRouter" in str(objectType):
                    G.ObjectInterruptionList.append(objectInterruption)

        from supervisor_class.simulator_class.manpy.simulation.ScheduledMaintenance import ScheduledMaintenance
        from supervisor_class.simulator_class.manpy.simulation.Failure import Failure
        from supervisor_class.simulator_class.manpy.simulation.PeriodicMaintenance import PeriodicMaintenance
        from supervisor_class.simulator_class.manpy.simulation.ShiftScheduler import ShiftScheduler
        from supervisor_class.simulator_class.manpy.simulation.ScheduledBreak import ScheduledBreak
        from supervisor_class.simulator_class.manpy.simulation.Break import Break

        for (element_id, element) in list(self.nodes.items()):
            element["id"] = element_id
            scheduledMaintenance = element.get("interruptions", {}).get(
                "scheduledMaintenance", {}
            )
            if len(scheduledMaintenance):
                start = float(scheduledMaintenance.get("start", 0))
                duration = float(scheduledMaintenance.get("duration", 1))
                victim = self.ObjectById(element["id"])
                SM = ScheduledMaintenance(victim=victim, start=start, duration=duration)
                G.ObjectInterruptionList.append(SM)
                G.ScheduledMaintenanceList.append(SM)
            failure = element.get("interruptions", {}).get("failure", None)
            if failure:
                victim = self.ObjectById(element["id"])
                deteriorationType = failure.get("deteriorationType", "constant")
                waitOnTie = failure.get("waitOnTie", False)
                F = Failure(
                    victim=victim,
                    distribution=failure,
                    repairman=victim.repairman,
                    deteriorationType=deteriorationType,
                    waitOnTie=waitOnTie,
                )
                G.ObjectInterruptionList.append(F)
                G.FailureList.append(F)
            periodicMaintenance = element.get("interruptions", {}).get(
                "periodicMaintenance", None
            )
            if periodicMaintenance:
                distributionType = periodicMaintenance.get("distributionType", "No")
                victim = self.ObjectById(element["id"])
                PM = PeriodicMaintenance(
                    victim=victim,
                    distribution=periodicMaintenance,
                    repairman=victim.repairman,
                )
                G.ObjectInterruptionList.append(PM)
                G.PeriodicMaintenanceList.append(PM)
            shift = element.get("interruptions", {}).get("shift", {})
            if len(shift):
                victim = self.ObjectById(element["id"])
                shiftPattern = list(shift.get("shiftPattern", []))
                for index, record in enumerate(shiftPattern):
                    if record is shiftPattern[-1]:
                        break
                    next = shiftPattern[index + 1]
                    if record[1] == next[0]:
                        record[1] = next[1]
                        shiftPattern.remove(next)
                endUnfinished = bool(int(shift.get("endUnfinished", 0)))
                receiveBeforeEndThreshold = float(shift.get("receiveBeforeEndThreshold", 0))
                thresholdTimeIsOnShift = bool(int(shift.get("thresholdTimeIsOnShift", 1)))
                rolling = bool(int(shift.get("rolling", 0)))
                lastOffShiftDuration = float(shift.get("lastOffShiftDuration", 10))
                SS = ShiftScheduler(
                    victim=victim,
                    shiftPattern=shiftPattern,
                    endUnfinished=endUnfinished,
                    receiveBeforeEndThreshold=receiveBeforeEndThreshold,
                    thresholdTimeIsOnShift=thresholdTimeIsOnShift,
                    rolling=rolling,
                    lastOffShiftDuration=lastOffShiftDuration,
                )
                G.ObjectInterruptionList.append(SS)
                G.ShiftSchedulerList.append(SS)
            scheduledBreak = element.get("interruptions", {}).get("scheduledBreak", None)
            if scheduledBreak:
                victim = self.ObjectById(element["id"])
                breakPattern = list(scheduledBreak.get("breakPattern", []))
                for index, record in enumerate(breakPattern):
                    if record is breakPattern[-1]:
                        break
                    next = breakPattern[index + 1]
                    if record[1] == next[0]:
                        record[1] = next[1]
                        shiftPattern.remove(next)
                endUnfinished = bool(int(scheduledBreak.get("endUnfinished", 0)))
                receiveBeforeEndThreshold = float(
                    scheduledBreak.get("receiveBeforeEndThreshold", 0)
                )
                rolling = bool(int(scheduledBreak.get("rolling", 0)))
                lastNoBreakDuration = float(scheduledBreak.get("lastOffShiftDuration", 10))
                SB = ScheduledBreak(
                    victim=victim,
                    breakPattern=breakPattern,
                    endUnfinished=endUnfinished,
                    receiveBeforeEndThreshold=receiveBeforeEndThreshold,
                    rolling=rolling,
                    lastNoBreakDuration=lastNoBreakDuration,
                )
                G.ObjectInterruptionList.append(SB)
                G.ShiftSchedulerList.append(SB)
            br = element.get("interruptions", {}).get("break", None)
            if br:
                victim = self.ObjectById(element["id"])
                endUnfinished = bool(int(br.get("endUnfinished", 1)))
                offShiftAnticipation = br.get("offShiftAnticipation", 0)
                BR = Break(
                    victim=victim,
                    distribution=br,
                    endUnfinished=endUnfinished,
                    offShiftAnticipation=offShiftAnticipation,
                )
                G.ObjectInterruptionList.append(BR)
                G.BreakList.append(BR)

    def createWIP(self):
        G.JobList = []
        G.WipList = []
        G.EntityList = []
        G.PartList = []
        G.OrderComponentList = []
        G.DesignList = []  # list of the OrderDesigns in the system
        G.OrderList = []
        G.MouldList = []
        G.BatchList = []
        G.SubBatchList = []
        G.CapacityEntityList = []
        G.CapacityProjectList = []
        G.pendingEntities = []

        self.EntityListC = []
        self.numOfWIP = 0

        if self.bom:
            orders = self.bom.get("productionOrders", [])
            for prodOrder in orders:
                orderClass = prodOrder.get("_class", None)
                orderType = Globals.getClassFromName(orderClass)
                if orderClass == "manpy.Order":
                    id = prodOrder.get("id", "not found")
                    name = prodOrder.get("name", "not found")
                    priority = int(prodOrder.get("priority", "0"))
                    dueDate = float(prodOrder.get("dueDate", "0"))
                    orderDate = float(prodOrder.get("orderDate", "0"))
                    isCritical = bool(int(prodOrder.get("isCritical", "0")))
                    componentsReadyForAssembly = bool(
                        (prodOrder.get("componentsReadyForAssembly", False))
                    )
                    componentsList = prodOrder.get("componentsList", {})
                    extraPropertyDict = {}
                    for key, value in list(prodOrder.items()):
                        if key not in ("_class", "id"):
                            extraPropertyDict[key] = value
                    O = Order(
                        "G" + id,
                        "general " + name,
                        route=[],
                        priority=priority,
                        dueDate=dueDate,
                        orderDate=orderDate,
                        isCritical=isCritical,
                        componentsList=componentsList,
                        componentsReadyForAssembly=componentsReadyForAssembly,
                        extraPropertyDict=extraPropertyDict,
                    )
                    G.OrderList.append(O)
                else:
                    productionOrderClass = prodOrder.get("_class", None)
                    productionOrderType = Globals.getClassFromName(productionOrderClass)
                    inputDict = dict(prodOrder)
                    inputDict.pop("_class")
                    from supervisor_class.simulator_class.manpy.simulation.Entity import Entity

                    if issubclass(productionOrderType, Entity):
                        entity = productionOrderType(**inputDict)
                        self.EntityListC.append(entity)

        for (element_id, element) in list(self.nodes.items()):
            element["id"] = element_id
            wip = element.get("wip", [])
            self.numOfWIP += len(wip)
            from supervisor_class.simulator_class.manpy.simulation.OrderDesign import OrderDesign

            for entity in wip:
                if self.bom:
                    if self.bom.get("productionOrders", []):
                        for order in G.OrderList:
                            if order.componentsList:
                                for componentDict in order.componentsList:
                                    if entity["id"] == componentDict["id"]:
                                        entityCurrentSeq = int(
                                            entity["sequence"]
                                        )
                                        entityRemainingProcessingTime = entity.get(
                                            "remainingProcessingTime", {}
                                        )
                                        entityRemainingSetupTime = entity.get(
                                            "remainingSetupTime", {}
                                        )
                                        ind = 0
                                        solution = False
                                        for i, step in enumerate(
                                            componentDict.get("route", [])
                                        ):
                                            stepSeq = step[
                                                "sequence"
                                            ]
                                            if stepSeq == "":
                                                stepSeq = 0
                                            if (
                                                int(stepSeq) == entityCurrentSeq
                                                and element["id"] in step["stationIdsList"]
                                            ):
                                                ind = i
                                                solution = True
                                                break
                                        assert solution, (
                                            "something is wrong with the initial step of "
                                            + entity["id"]
                                        )
                                        entityRoute = componentDict.get("route", [])[ind:]
                                        entity = dict(componentDict)
                                        entity.pop("route")
                                        entity[
                                            "route"
                                        ] = entityRoute
                                        entity["order"] = order.id
                                        entity[
                                            "remainingProcessingTime"
                                        ] = entityRemainingProcessingTime
                                        entity[
                                            "remainingSetupTime"
                                        ] = entityRemainingSetupTime
                                        break

                entityClass = entity.get("_class", None)
                entityType = Globals.getClassFromName(entityClass)
                inputDict = dict(entity)
                inputDict.pop("_class")
                from supervisor_class.simulator_class.manpy.simulation.Entity import Entity

                if issubclass(entityType, Entity) and (not entityClass == "manpy.Order"):
                    if entity.get("order", None):
                        entityOrder = self.ObjectById(entity["order"])
                        inputDict.pop("order")
                        entity = entityType(order=entityOrder, **inputDict)
                        entity.routeInBOM = True
                    else:
                        entity = entityType(**inputDict)
                    self.EntityListC.append(entity)
                    object = self.ObjectById(element["id"])
                    entity.currentStation = object

                if entityClass == "manpy.Order":
                    id = entity.get("id", "not found")
                    name = entity.get("name", "not found")
                    priority = int(entity.get("priority", "0"))
                    dueDate = float(entity.get("dueDate", "0"))
                    orderDate = float(entity.get("orderDate", "0"))
                    isCritical = bool(int(entity.get("isCritical", "0")))
                    basicsEnded = bool(int(entity.get("basicsEnded", "0")))
                    componentsReadyForAssembly = bool(
                        (entity.get("componentsReadyForAssembly", False))
                    )
                    manager = entity.get("manager", None)
                    if manager:
                        for operator in G.OperatorsList:
                            if manager == operator.id:
                                manager = operator
                                break
                    componentsList = entity.get("componentsList", {})
                    JSONRoute = entity.get(
                        "route", []
                    )
                    route = [x for x in JSONRoute]

                    extraPropertyDict = {}
                    for key, value in list(entity.items()):
                        if key not in ("_class", "id"):
                            extraPropertyDict[key] = value

                    odAssigned = False
                    for element in route:
                        elementIds = element.get("stationIdsList", [])
                        for obj in self.ObjectList:
                            for elementId in elementIds:
                                if obj.id == elementId and obj.type == "OrderDecomposition":
                                    odAssigned = True
                    if not odAssigned:
                        odId = None
                        for obj in self.ObjectList:
                            if obj.type == "OrderDecomposition":
                                odId = obj.id
                                break
                        if odId:
                            route.append(
                                {
                                    "stationIdsList": [odId],
                                    "processingTime": {
                                        "distributionType": "Fixed",
                                        "mean": "0",
                                    },
                                }
                            )
                    O = Order(
                        "G" + id,
                        "general " + name,
                        route=[],
                        priority=priority,
                        dueDate=dueDate,
                        orderDate=orderDate,
                        isCritical=isCritical,
                        basicsEnded=basicsEnded,
                        manager=manager,
                        componentsList=componentsList,
                        componentsReadyForAssembly=componentsReadyForAssembly,
                        extraPropertyDict=extraPropertyDict,
                    )
                    OD = OrderDesign(
                        id,
                        name,
                        route,
                        priority=priority,
                        dueDate=dueDate,
                        orderDate=orderDate,
                        isCritical=isCritical,
                        order=O,
                        extraPropertyDict=extraPropertyDict,
                    )
                    G.OrderList.append(O)
                    G.OrderComponentList.append(OD)
                    G.DesignList.append(OD)
                    G.WipList.append(OD)
                    self.EntityListC.append(OD)
                    G.JobList.append(OD)

        if G.initializingFlag and G.initializingFilename != "":
            self.initializeWIP_fromFile()

    def initializeWIP_fromFile(self):
        initializingFileName = G.initializingFilename
        initializingFile = open(initializingFileName, "r")
        initializingContent = initializingFile.read()
        initializingFile.close()
        initializingList = initializingContent.split("\n")
        entityClass = "manpy.Part"
        entityType = Globals.getClassFromName(entityClass)
        self.numOfWIP += len(initializingList)
        for i in range(len(initializingList)):
            PartID = "P" + str(int(i) + 1)
            PartName = "Part" + str(int(i) + 1)
            PartDict = {
                "id": PartID,
                "name": PartName
            }
            entityR = entityType(**PartDict)
            initializingQueueName = self.ObjectById("Q" + str(int(initializingList[i])))
            self.EntityListC.append(entityR)
            entityR.currentStation = initializingQueueName

    def initializeWIP(self, init_list):
        initializingList = init_list
        entityClass = "manpy.Part"
        entityType = Globals.getClassFromName(entityClass)
        self.numOfWIP += len(initializingList)
        for i in range(len(initializingList)):
            PartID = "P" + str(int(i) + 1)
            PartName = "Part" + str(int(i) + 1)
            PartDict = {
                "id": PartID,
                "name": PartName
            }
            entityR = entityType(**PartDict)
            initializingQueueName = self.ObjectById(
                "Q" + str(int(initializingList[i])))
            self.EntityListC.append(entityR)
            entityR.currentStation = initializingQueueName

    def ObjectById(self, object_id):
        for obj in (
            self.ObjectList
            + G.ObjectResourceList
            + self.EntityListC
            + G.ObjectInterruptionList
            + G.OrderList
        ):
            if obj.id == object_id:
                return obj
        return None

    def setupWIP(self, entityList):
        for entity in entityList:
            if entity.type in ["Part", "Batch", "SubBatch", "CapacityEntity", "Vehicle"]:
                if entity.currentStation:
                    obj = entity.currentStation
                    obj.getActiveObjectQueue().append(
                        entity
                    )
                    entity.schedule.append(
                        {"station": obj, "entranceTime": self.Thisenv.now}
                    )

            elif entity.type in ["Job", "OrderComponent", "Order", "OrderDesign", "Mould"]:
                currentObjectIds = entity.remainingRoute[0].get("stationIdsList", [])

                objectId = currentObjectIds[0]
                obj = self.ObjectById(objectId)
                obj.getActiveObjectQueue().append(
                    entity
                )
                if obj.__class__.__name__ == "MouldAssemblyBuffer":
                    entity.readyForAssembly = 1

                nextObjectIds = entity.remainingRoute[1].get("stationIdsList", [])
                nextObjects = []
                for nextObjectId in nextObjectIds:
                    nextObject = self.ObjectById(nextObjectId)
                    nextObjects.append(nextObject)
                for nextObject in nextObjects:
                    if nextObject not in obj.next:
                        obj.next.append(nextObject)
                entity.currentStep = entity.remainingRoute.pop(
                    0
                )
                entity.schedule.append(
                    {"station": obj, "entranceTime": self.Thisenv.now}
                )
                if entity.currentStep:
                    if entity.currentStep.get("task_id", None):
                        entity.schedule[-1]["task_id"] = entity.currentStep["task_id"]
            if (not (entity.currentStation in G.MachineList)) and entity.currentStation:
                G.pendingEntities.append(entity)

            from supervisor_class.simulator_class.manpy.simulation.Queue import Queue

            if entity.currentStation:
                if issubclass(entity.currentStation.__class__, Queue):
                    if not entity.currentStation.canDispose.triggered:
                        if entity.currentStation.expectedSignals["canDispose"]:
                            succeedTuple = (self.Thisenv, self.Thisenv.now)
                            entity.currentStation.canDispose.succeed(succeedTuple)
                            entity.currentStation.expectedSignals["canDispose"] = 0
            if self.Thisenv.now == 0 and entity.currentStation:
                if entity.currentStation.class_name:
                    stationClass = entity.currentStation.__class__.__name__
                    if stationClass in [
                        "ProductionPoint",
                        "ConveyorMachine",
                        "ConveyorPoint",
                        "ConditionalPoint",
                        "Machine",
                        "BatchScrapMachine",
                        "MachineJobShop",
                        "BatchDecomposition",
                        "BatchReassembly",
                        "M3",
                        "MouldAssembly",
                        "BatchReassemblyBlocking",
                        "BatchDecompositionBlocking",
                        "BatchScrapMachineAfterDecompose",
                        "BatchDecompositionStartTime",
                    ]:
                        entity.currentStation.currentEntity = entity
                        if not (entity.currentStation.initialWIP.triggered):
                            if entity.currentStation.expectedSignals["initialWIP"]:
                                succeedTuple = (self.Thisenv, self.Thisenv.now)
                                entity.currentStation.initialWIP.succeed(succeedTuple)
                                entity.currentStation.expectedSignals["initialWIP"] = 0

    def initializeFromList(self, processingTable):
        MachList = processingTable.columns.tolist()
        for mach in MachList:
            procTimes = (processingTable[mach].dropna()).tolist()
            condition = False
            for obj in self.ObjectList:
                if mach == obj.id:
                    condition = True
                    obj.procTimeVal = ProcessingTimeList(None, procTimes)
                    obj.fromListFlag = True
            if not condition:
                print(f'ERROR: it exist no object named {mach!r}, the columns headers should be named like the object'
                      f' they refer to!!!')

    def initializeDistributions(self, distributionTable):
        MachList = distributionTable.columns.tolist()
        for mach in MachList:
            distr_name = distributionTable[mach][0]
            distr_param = distributionTable[mach][1]
            condition = False
            for obj in self.ObjectList:
                if mach == obj.id:
                    condition = True
                    processingTime_temp = Globals.convertDistribution(distr_name, distr_param)
                    processingTime = obj.getOperationTime(time=processingTime_temp)
                    obj.rng = RandomNumberGenerator(obj, processingTime)
            if not condition:
                print(f'ERROR: it exist no object named {mach!r}, the columns headers should be named like the object'
                      f' they refer to!!!')

    def setTopology(self):
        for element in self.ObjectList:
            next = []
            previous = []
            for j in range(len(element.previousIds)):
                for q in range(len(self.ObjectList)):
                    if self.ObjectList[q].id == element.previousIds[j]:
                        previous.append(self.ObjectList[q])

            for j in range(len(element.nextIds)):
                for q in range(len(self.ObjectList)):
                    if self.ObjectList[q].id == element.nextIds[j]:
                        next.append(self.ObjectList[q])

            if element.type == "Source":
                element.defineRouting(next)
            elif element.type == "Exit":
                element.defineRouting(previous)
            elif element.type == "Dismantle":
                nextPart = []
                nextFrame = []
                for j in range(len(element.nextPartIds)):
                    for q in range(len(self.ObjectList)):
                        if self.ObjectList[q].id == element.nextPartIds[j]:
                            nextPart.append(self.ObjectList[q])
                for j in range(len(element.nextFrameIds)):
                    for q in range(len(self.ObjectList)):
                        if self.ObjectList[q].id == element.nextFrameIds[j]:
                            nextFrame.append(self.ObjectList[q])
                element.defineRouting(previous, next)
                element.definePartFrameRouting(nextPart, nextFrame)
            else:
                element.defineRouting(previous, next)

    def initializeObjects(self):
        for element in (
            self.ObjectList
            + G.ObjectResourceList
            + self.EntityListC
            + G.ObjectInterruptionList
            + G.RouterList
        ):
            if element.name in [
                "Machine",
                "Transport",
                "Queue",
                "Source",
                "Exit",
            ]:
                element.initialize(self.Thisenv)
            else:
                element.initialize()

    def activateObjects(self):
        for element in G.ObjectInterruptionList:
            self.Thisenv.process(element.run())
        for element in self.ObjectList:
            self.Thisenv.process(element.run())

    def runTraceSimulation(self, proc_time, init_list=None):

        self.ObjectList = []
        self.RouterListC = []
        self.EntityListC = []

        G.ObjList = []
        G.RouterList = []
        self.createObjectResourcesAndCoreObjects()
        self.createObjectInterruptions()
        self.setTopology()

        G.RouterList = self.RouterListC
        G.calculateEventList = True
        G.MachineEventList = [[], [], [], []]
        G.calculateFinalPosition = True
        G.FinalPositionList = []

        start = time.time()

        maxSimTime = 100000
        G.trace = "No"
        G.console = "No"
        G.confidenceLevel = "0.95"
        G.seed = 1
        G.extraPropertyDict = {}
        G.initializingFlag = False
        G.initializingFilename = ""

        self.initializeFromList(proc_time)

        self.Thisenv = simpy.Environment()
        if G.RouterList:
            G.RouterList[0].isActivated = False
            G.RouterList[0].isInitialized = False

        G.Rnd = Random()
        G.numpyRnd.random.seed()

        self.createWIP()
        if init_list:
            self.initializeWIP(init_list)
        self.initializeObjects()
        self.setupWIP(self.EntityListC)
        self.activateObjects()

        self.Thisenv.run(until=maxSimTime)

        maxSimTime_temp = G.MachineEventList[0][-1]
        postProcCorr = True

        for element in self.ObjectList + G.ObjectResourceList + G.RouterList:
            element.postProcessing(MaxSimtime=maxSimTime_temp, correction=postProcCorr)

        PrintRoute.outputRoute()

        if G.trace == "Yes":
            ExcelHandler.outputTrace("trace")
            import io

            traceStringIO = io.StringIO()
            G.traceFile.save(traceStringIO)
            encodedTrace = traceStringIO.getvalue().encode("base64")
            ExcelHandler.resetTrace()

        outputDict = {}
        outputDict["_class"] = "manpy.Simulation"
        outputDict["general"] = {}
        outputDict["general"]["_class"] = "manpy.Configuration"
        outputDict["general"]["totalExecutionTime"] = time.time() - start
        outputDict["elementList"] = []

        for obj in self.ObjectList + G.RouterList:
            outputDict["elementList"].append(obj.outputResultsDict())

        if G.trace == "Yes":
            jsonTRACE = {
                "_class": "manpy.Simulation",
                "id": "TraceFile",
                "results": {"trace": encodedTrace},
            }
            G.outputJSON["elementList"].append(jsonTRACE)

        outputDict.update({'eventlog': G.MachineEventList})
        G.FinalPositionList.sort()
        outputDict.update({'final_position': G.FinalPositionList})
        del outputDict['elementList'][0]['results']['system_time_trace'][0][self.numOfWIP]

        return outputDict

    def runStochSimulation(self, distrib_table=None, sim_time=10, n_replications=1, init_list=None, seed=1):

        self.ObjectList = []
        self.RouterListC = []
        self.EntityListC = []

        G.ObjList = []
        G.RouterList = []
        self.createObjectResourcesAndCoreObjects()
        self.createObjectInterruptions()
        self.setTopology()

        G.RouterList = self.RouterListC
        G.calculateEventList = True
        G.MachineEventList = [[], [], [], []]
        G.calculateFinalPosition = False

        start = time.time()

        G.trace = "No"
        G.console = "No"
        G.confidenceLevel = "0.95"
        G.seed = seed
        G.extraPropertyDict = {}
        G.initializingFlag = False
        G.initializingFilename = ""

        if distrib_table is not None:
            self.initializeDistributions(distrib_table)

        for i in range(n_replications):
            self.Thisenv = simpy.Environment()
            sim_time_fin = sim_time

            if G.RouterList:
                G.RouterList[0].isActivated = False
                G.RouterList[0].isInitialized = False

            if G.seed:
                G.Rnd = Random("%s%s" % (G.seed, i))
                G.numpyRnd.random.seed(G.seed + i)
            else:
                G.Rnd = Random()
                G.numpyRnd.random.seed()
            self.createWIP()
            if init_list:
                self.initializeWIP(init_list)
            self.initializeObjects()
            self.setupWIP(self.EntityListC)
            self.activateObjects()

            if G.maxSimTime == -1:
                self.Thisenv.run(until=float("inf"))

                endList = []
                for exit in G.ExitList:
                    endList.append(exit.timeLastEntityLeft)

                if float(max(endList)) != 0 and (
                    self.Thisenv.now == float("inf") or self.Thisenv.now == max(endList)
                ):
                    sim_time_fin = float(max(endList))
                else:
                    print("simulation ran for 0 time, something may have gone wrong")
                    logger.info("simulation ran for 0 time, something may have gone wrong")
            else:
                self.Thisenv.run(until=sim_time_fin)

            for element in self.ObjectList + G.ObjectResourceList + G.RouterList:
                element.postProcessing(MaxSimtime=sim_time_fin)

            PrintRoute.outputRoute()

            if G.trace == "Yes":
                ExcelHandler.outputTrace("trace" + str(i))
                import io

                traceStringIO = io.StringIO()
                G.traceFile.save(traceStringIO)
                encodedTrace = traceStringIO.getvalue().encode("base64")
                ExcelHandler.resetTrace()

        outputDict = {}
        outputDict["_class"] = "manpy.Simulation"
        outputDict["general"] = {}
        outputDict["general"]["_class"] = "manpy.Configuration"
        outputDict["general"]["totalExecutionTime"] = time.time() - start
        outputDict["elementList"] = []

        for obj in self.ObjectList + G.RouterList:
            outputDict["elementList"].append(obj.outputResultsDict())

        for i in range(n_replications):
            outputDict['elementList'][0]['results']['interarrival_trace'][i] = \
                outputDict['elementList'][0]['results']['interarrival_trace'][i][self.numOfWIP:]
            outputDict['elementList'][0]['results']['system_time_trace'][i] = \
                outputDict['elementList'][0]['results']['system_time_trace'][i][self.numOfWIP:]

        if G.trace == "Yes":
            jsonTRACE = {
                "_class": "manpy.Simulation",
                "id": "TraceFile",
                "results": {"trace": encodedTrace},
            }
            G.outputJSON["elementList"].append(jsonTRACE)

        outputDict.update({'eventlog': G.MachineEventList})

        return outputDict
