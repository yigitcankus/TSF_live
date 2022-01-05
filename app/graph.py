# Function to calculate the sum of array
class DataArray(object):
   
    def LogicalInfo(self, data, error_rates):
        new_dict= {}
        for i in range(len(data)):
                new_dict[data[i]] = error_rates[i]
        return new_dict

    def GraphInfo(self, graph_data):     
        
        #print("here is the resulted graph:", graph_data, type(graph_data))

        #print("here is the error rates of the next dates:")
        #to print the err rate for each data
        
        return graph_data


def deneme():   
    read_from_csv = [(1,2), (2,3), (3,2), (4,5)]
            
            #the next x days
    data=[5,6,7,8,9,10,11,12,13,14,15]

            #values of next x days
    value=[2,1,3,2,3,8,5,2,4,0,1]

            #error rates of predictions
    error_rates = [0.5, 0.6, 0.6, 0.9, 0.5, 0.6, 0.6, 0.9, 0.5, 0.6, 0.6]

            #appends the graph data with the prediction values
    for i in range(len(value)):
        t1 = tuple((data[i], value[i]))
        read_from_csv.append(t1)


    temp = DataArray()
            
    res = temp.GraphInfo(read_from_csv)
    
    return res

def logicalInformationGraph():
        #values of next x days
    value=[2,1,3,2,3,8,5,2,4,0,1]

            #error rates of predictions
    error_rates = [0.5, 0.6, 0.6, 0.9, 0.5, 0.6, 0.6, 0.9, 0.5, 0.6, 0.6]

    temp = DataArray()

    res= temp.LogicalInfo(value,error_rates)
   
    return error_rates
        

  
    


        

        
        

        

