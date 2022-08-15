/**
 * 
 * Copyright (c) 2020, Jose-Luis Lisani, joseluis.lisani@uib.es
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <vector>
#include <algorithm>

//get list of minima of a histogram
//the left and right endpoints are included in the list
std::vector<int> get_minima(float *hist, int size)
{
    std::vector<int> minpos;

    //type of each point (bin) in histogram: 
    //1: minimum (value < previous, value < next)
    //2: potential left endpoint of flat minimum (value < previous, value = next)
    //3: potential right endpoint of flat minimum (value < next, value = previous)
    //4: flat point (value=previous=next)
    //5: endpoint of histogram
    //0: rest of cases
    float diffprev, diffnext; 
    int *type=new int[size];
    type[0]=5;
    type[size-1]=5;
    for (int i=1; i < size-1; i++) {
        type[i]=0;
        diffprev=hist[i]-hist[i-1];
        diffnext=hist[i]-hist[i+1];
        if ((diffprev < 0) && (diffnext < 0)) type[i]=1; //minimum
        if ((diffprev == 0) && (diffnext == 0)) type[i]=4; //flat
        if ((diffprev < 0) && (diffnext == 0)) type[i]=2; //potential left endpoint of flat minimum
        if ((diffprev == 0) && (diffnext < 0)) type[i]=3; //potential right endpoint of flat minimum 
    }
    //check flat minima
    for (int i=1; i < size-1; i++) {
        if (type[i] == 2) { //potential left endpoint of flat minimum
            //look for right endpoint
            int j;
            for (j=i+1; (j < size-1) && (type[j] == 4); j++);
            if (type[j] == 3) { //found right endpoint
                //mark center of flat zone as minimum
                type[(i+j)/2]=1;
            }
        }
    }

    //output list of minima + endpoints
    minpos.push_back(0); //left endpoint
    for (int i=1; i < size-1; i++) {
        if (type[i] == 1) minpos.push_back(i); //minimum
    }
    minpos.push_back(size-1); //right endpoint
    
    delete[] type;
    return minpos;
}

//get list of maxima of a histogram
std::vector<int> get_maxima(float *hist, int size)
{
    std::vector<int> maxpos;

    //type of each point (bin) in histogram: 
    //1: maximum (value > previous, value > next)
    //2: potential left endpoint of flat maximum (value > previous, value = next)
    //3: potential right endpoint of flat maximum (value > next, value = previous)
    //4: flat point (value=previous=next)
    //0: rest of cases
    float diffprev, diffnext; 
    int *type=new int[size];
    //check all except endpoints
    for (int i=1; i < size-1; i++) {
        type[i]=0;
        diffprev=hist[i]-hist[i-1];
        diffnext=hist[i]-hist[i+1];
        if ((diffprev > 0) && (diffnext > 0)) type[i]=1; //maximum
        if ((diffprev == 0) && (diffnext == 0)) type[i]=4; //flat
        if ((diffprev > 0) && (diffnext == 0)) type[i]=2; //potential left endpoint of flat maximum
        if ((diffprev == 0) && (diffnext > 0)) type[i]=3; //potential right endpoint of flat maximum 
    }
    //check endpoints
    type[0]=0;
    type[size-1]=0;
    if (hist[0] > hist[1]) type[0]=1; //maximum
    if (hist[0] == hist[1]) type[0]=2; //potential left endpoint of flat maximum
    if (hist[size-1] > hist[size-2]) type[size-1]=1; //maximum
    if (hist[size-1] == hist[size-2]) type[size-1]=3; //potential right endpoint of flat maximum 
    
    //check flat maximum
    for (int i=0; i < size; i++) {
        if (type[i] == 2) { //potential left endpoint of flat maximum
            //look for right endpoint
            int j;
            for (j=i+1; (j < size-1) && (type[j] == 4); j++);
            if (type[j] == 3) { //found right endpoint
                //mark center of flat zone as maximum
                type[(i+j)/2]=1;
            }
        }
    }

    //output list of maxima 
    for (int i=0; i < size; i++) {
        if (type[i] == 1) maxpos.push_back(i); //maximum
    }
     
    delete[] type;
    return maxpos;
}



//find monotonically decreasing and increasing intervals of the histogram
//type of each point (bin) in histogram: 
//0: flat (current = previous)
//1: increasing (current > previous)
//2: decreasing (current < previous)
//This function implements lines 3-6 of the Pool Adjacent Violators algorithm 
//(Algorithm 3, in accompanying paper)
void get_monotone_info(float *hist, int size, int *type, 
                       int &nincreasing, int &ndecreasing,
                       unsigned char extend_increasing)
{
    nincreasing=0;
    ndecreasing=0;
    
	float diffprev;
	//assume first bin = flat
	type[0]=0;
	for (int i=1; i < size; i++) {
	    diffprev=hist[i]-hist[i-1];
	    type[i]=0;
	    if (diffprev > 0) {
	    	type[i]=1;
	    	nincreasing++;
	    } else {
	    	if (diffprev < 0) {
	    	    type[i]=2;
	    	    ndecreasing++;
	    	}
	    }
	}

	//extend strict monotony type (< or >) to general monotony (<= or >=)
	int typeV;
    if (extend_increasing) typeV=1;
    else typeV=2;
	
	//extend to the left of non-flat bin
	int ilast=-1;
	for (int i=1; i < size; i++) {
	    if (type[i] == typeV) { //non-flat bin
	        for (int j=i-1; (j > 0) && (type[j] == 0); j--) type[j]=typeV;
	        ilast=i;
	    }
	}
	
    //last non-flat bin: extend to the right
    if (ilast != -1) for (int j=ilast+1; (j < size) && (type[j] == 0); j++) type[j]=typeV;
    

}

//replace a monotonically increasing (resp. decreasing) interval of the 
//histogram by a constant value
//This code implements lines 7-9 of the Pool Adjacent Violators algorithm 
//(Algorithm 3, in accompanying paper)
void replace_monotone(float *hist, int size, int *type, 
                      unsigned char replace_increasing)
{    
	int typeV;
    if (replace_increasing) typeV=1;
    else typeV=2;
       
    //find monotonically decreasing/increasing intervals
    int i=0;
    int istart, iend;
    unsigned char isfirst=1; //flag that indicates beginning of interval
    while (i < size) {
        if (type[i] == typeV) {
            if (isfirst) {
                //istart=i;
                istart=(i > 0)?(i-1):i; //take value to the left of left endpoint of interval
                isfirst=0;
            } 
            if ((i == size-1) || (type[i+1] != typeV)) {
                iend=i;
                //assign constant value to interval
                float C=0;
                for (int j=istart; j <= iend; j++) C+=hist[j];
                C/=(iend-istart+1);
                for (int j=istart; j <= iend; j++) hist[j]=C;
                isfirst=1;
            }                
        }
        i++;
    }
    
}

//Implement Pool Adjacent Violators algorithm 
//(Algorithm 3, in accompanying paper)
float *pool_adjacent_violators(float *hist, int size, unsigned char increasing)
{
    float *hmono=new float[size];
    //initialize with input histogram
    memcpy(hmono, hist, size*sizeof(float));

    //type of each point (bin) in histogram: 
    //0: flat (current = previous)
    //1: increasing (current > previous)
    //2: decreasing (current < previous)
    int *type=new int[size];
    int nincreasing, ndecreasing;
          
    if (increasing) { //get increasing histogram
        do {
            //find monotonically decreasing intervals of the histogram
            get_monotone_info(hmono, size, type, nincreasing, ndecreasing, 0);
            //replace monotonically decreasing intervals by a constant value 
            if (ndecreasing > 0) {
                replace_monotone(hmono, size, type, 0);
            }
        } while (ndecreasing > 0);//stop when no more decreasing intervals exist    
    } else {    //get decreasing histogram
        do {
            //find monotonically increasing intervals of the histogram
            get_monotone_info(hmono, size, type, nincreasing, ndecreasing, 1);
            //replace monotonically decreasing intervals by a constant value 
            if (nincreasing > 0) {
                replace_monotone(hmono, size, type, 1);
            }
        } while (nincreasing > 0);//stop when no more increasing intervals exist   
    }

    delete[] type;
    return hmono;
}

//compute relative entropy value
double relative_entropy(double r, double p)
{
    double H;
    if (r == 0.0) H=-log(1.0-p);
    else if (r == 1.0) H=-log(p);
    else H=(r*log(r/p)+(1.0-r)*log((1.0-r)/(1.0-p)));
    
    return H;
}

//compute cost (maximum relative entropy) associated to an interval of the 
//histogram, with respect to the monotone hypothesis
//This function implements Algorithm 2 in accompanying paper
double cost_monotone(float *hist0, int i1, int i2, unsigned char increasing, double logeps)
{
    //double logeps=0; //logeps=log(NFAmax)  
    //NFAmax=maximum expected number of false positives w.r.t. null hypothesis (monotony)
    
    //get subhistogram
    int L=i2-i1+1;
    float *hist=new float[L];
    for (int i=0; i < L; i++) hist[i]=hist0[i1+i];
           
    //get monotone estimation
    float *hmono=pool_adjacent_violators(hist, L, increasing);
            
    //Compute cost
    
    //cumulated histograms
    for (int i=1; i < L; i++) hist[i]+=hist[i-1];
    for (int i=1; i < L; i++) hmono[i]+=hmono[i-1];
    //meaningfullness threshold
    int N=hist[L-1];
    double threshold=(log((double)L*(L+1)/2)- logeps)/(double) N; 
        
    //find interval that more rejects the null hypothesis (monotony)
    //i.e. highest Kullblack-Leibler distance from hist to hmono
    double r, p, H, Hmax;
    for (int i=0; i < L ; i++)
        for(int j=i; j < L; j++) {
            //r: proportion of values in [i, j], for hist 
            if (i == 0) r=(double) hist[j];
            else r=(double) (hist[j]-hist[i-1]);
            r=r/(double) N;
            //p: proportion of values in [i, j], for hmono            
            if (i == 0) p=(double) hmono[j];
            else p=(double) (hmono[j]-hmono[i-1]);
            p=p/(double)N;
            //relative entropy (Kullblack-Leibler distance from hist to hmono)
            H=relative_entropy(r, p);
            if (((i == 0) && (j == 0)) || (H > Hmax)) Hmax=H;
            
        }
    
    //cost
    double cost= (double) N * Hmax - (log((double)L*(L+1)/2)- logeps);

//    printf("        Hmax=%e   N=%i  L=%i  cost=%e\n", Hmax, N, L, cost);
                
    delete[] hist;
    delete[] hmono;
    

    return cost;
}

//data structure to store the cost of merging intervals of the histogram
struct costdata {
    double cost;
    int imin1, imin2;
    int typemerging;
};



//each mode is composed of:
// minimum - maximum - minimum
// we want to check the if the unimodal hypothesis is preserved by merging
// minimumA - maximumA - ... - maximumB - minimumB
// maximumA is the maximum to the right of minimumA
// maximumB is the maximum to the left of minimumB
// i1 is the index of minimumA in separators list
// i2 is the index of minimumB in separators list
// i1 is the index of maximumA in maxima list
// i2-1 is the index of maximumB in maxima list
// the intervals can be merged if 
// 1) minimumA-maximumB is monotonically increasing (typemerging = 1)
// or
// 2) maximumA-minimumB is monotonically decreasing (typemerging = 2)
// If 1) then minimumA-maximumB-minimumB is unimodal 
//       and maximumA-... can be removed from the list of minima/maxima
// If 2) then minimumA-maximumA-minimumB is unimodal 
//       and  ...-maximumB can be removed from the list of minima/maxima
struct costdata cost_merging(float *hist, std::vector<costdata> &listcosts, 
                             std::vector<int> &separators, std::vector<int> &maxima,
                             int i1, int i2, double logeps)
{
    struct costdata cdata;
    
    //maximumB == maximum at position imin2-1
    double cost1=cost_monotone(hist, separators[i1], maxima[i2-1], 1, logeps); //increasing
    double cost2=cost_monotone(hist, maxima[i1], separators[i2], 0, logeps); //decreasing
    double cost;
    //keep the smallest
    if (cost1 < cost2) {
        cdata.cost=cost1;
        cdata.typemerging=1;
    } else {
        cdata.cost=cost2;
        cdata.typemerging=2;    
    }
    cdata.imin1=separators[i1];
    cdata.imin2=separators[i2];   
    
//    if ((cdata.imin1 == 194) && (cdata.imin2 == 239)) 
//        printf("    cost1(%i-%i) =%2.2f      cost2(%i-%i)=%2.2f\n", 
//                separators[i1], maxima[i2-1], cost1, 
//                maxima[i1], separators[i2], cost2);
         
    listcosts.push_back(cdata);
    
    return cdata;
}

//check if the cost of merging two intervals has been already computed
unsigned char cost_already_computed(std::vector<costdata> &listcosts, 
                                    int imin1, int imin2, struct costdata &cdata)
{
    unsigned char found=0;
    for (int k=0; k < listcosts.size() && !found; k++) {
        if ((listcosts[k].imin1 == imin1) && (listcosts[k].imin2 == imin2)) {
            found=1;
            cdata=listcosts[k];
        }
    }
    
    return found;
}

//Main algorithm
//This function implements the FTC algorithm 
//(Algorithm 1 in accompanying paper, if option circularhist == 0)
//(Algorithm 4 in accompanying paper, if option circularhist == 1)
std::vector<int> FTCsegmentation(float *histIn, int sizeIn, float eps, 
                            unsigned char circularhist)
{   
    float *hist;
    int size;
    
    double logeps=log(eps); //eps=NFAmax=maximum expected number of false alarms
    
    if (circularhist) {
        //if circular histogram then triplicate it, compute separators, and
        //output results for central copy
        size=3*sizeIn;
        hist=new float[size];
        for (int i=0; i < sizeIn; i++) {
            hist[i]=histIn[i];
            hist[i+sizeIn]=histIn[i];
            hist[i+2*sizeIn]=histIn[i];
        }
    } else {
        hist=histIn;
        size=sizeIn;
    }

    //get initial separators: position of minima of histogram + endpoints
    std::vector<int> separators;
    separators=get_minima(hist, size);
           
    std::vector<int> maxima;
    maxima=get_maxima(hist, size);
    
    //check maxima
    int nmax=maxima.size();    
    
    //number of initial separators
    int n=separators.size();
    //number of intervals
    int nintervals=n-1;
    
    //printf("%i minima, %i maxima, %i intervals\n", n, nmax, nintervals);    
    
//    printf("Minima:\n");
//    for (int k=0; k < n; k++) printf("%i %i\n", k, separators[k]);
//    printf("Maxima:\n");
//    for (int k=0; k < nmax; k++) printf("%i %i\n", k, maxima[k]);
    

    std::vector<costdata> listcosts;
    
    //Try merging intervals 1+j consecutive intervals (current + j consecutive)
    int j=1;
   
    while (nintervals > j) {
    
        //merging flag: initially set to 1
        unsigned char do_merging=1;
        while (do_merging && (nintervals > j)) {        
                     
            //compute cost of merging each interval with its successive interval
            //and find lowest cost
            struct costdata cdata, cdatalowest;
            int ilowest=-1;
            for (int i=0; i < nintervals-j; i++) {
            
                //in case of circular histogram do not try merging if the interval
                //is larger than the original input histogram
                if (circularhist && (separators[i+j+1]-separators[i] > sizeIn)) continue;
                
                if (!cost_already_computed(listcosts, separators[i], separators[i+j+1], cdata)) {
                    cdata=cost_merging(hist, listcosts, separators, maxima, i, i+j+1, logeps);  
                    //printf("    Cost merging %i-%i: %2.2f\n", separators[i], separators[i+j+1], cdata.cost);             
                }
                if ((ilowest == -1) || (cdata.cost < cdatalowest.cost)) {
                    cdatalowest=cdata;
                    ilowest=i;
                }
            }
            if (cdatalowest.cost > 0)  printf("Merge %i-%i   Cost=%2.2f\n", cdatalowest.imin1, cdatalowest.imin2, cdatalowest.cost);      
            //merge intervals with lowest cost, if it is smaller than 0
            if ((ilowest != -1) && (cdatalowest.cost < 10000)) {

//                printf("Merge %i-%i   Cost=%2.2f\n", cdatalowest.imin1, cdatalowest.imin2, cdatalowest.cost);
                //remove minima with index ilowest+1 to ilowest+j
                separators.erase(separators.begin() + ilowest+1, separators.begin() + ilowest+j+1);                
                //remove maxima associated to the removed minima
                if (cdatalowest.typemerging == 1) {
                    maxima.erase(maxima.begin() + ilowest, maxima.begin() + ilowest+j);
                }
                if (cdatalowest.typemerging == 2) {
                    maxima.erase(maxima.begin() + ilowest+1, maxima.begin() + ilowest+j+1);
                }
                n=separators.size();
                nintervals=n-1;
            } else do_merging=0; //stop merging
            
        }
    
        j++; //increase number of intervals to merge
    }
   
    if (circularhist) {
        //output results for central copy
        std::vector<int> separatorsC;
        for (int k=0; k < separators.size(); k++) {
            if ((separators[k] >= sizeIn) && (separators[k] < 2*sizeIn)) {
                separatorsC.push_back(separators[k]-sizeIn);
            }
        }
        //if no separators then add endpoints
        if (separatorsC.size() == 0) {
            separatorsC.push_back(0);
            separatorsC.push_back(sizeIn-1);                    
        }
        delete[] hist;
        return separatorsC;
    } 
    
    return separators;
}

std::vector<int> FTCsegmentation(float *histIn, int sizeIn, int cost_thresh, float eps, 
                            unsigned char circularhist)
{   
    float *hist;
    int size;
    
    double logeps=log(eps); //eps=NFAmax=maximum expected number of false alarms
    
    if (circularhist) {
        //if circular histogram then triplicate it, compute separators, and
        //output results for central copy
        size=3*sizeIn;
        hist=new float[size];
        for (int i=0; i < sizeIn; i++) {
            hist[i]=histIn[i];
            hist[i+sizeIn]=histIn[i];
            hist[i+2*sizeIn]=histIn[i];
        }
    } else {
        hist=histIn;
        size=sizeIn;
    }

    //get initial separators: position of minima of histogram + endpoints
    std::vector<int> separators;
    separators=get_minima(hist, size);
           
    std::vector<int> maxima;
    maxima=get_maxima(hist, size);
    
    //check maxima
    int nmax=maxima.size();    
    
    //number of initial separators
    int n=separators.size();
    //number of intervals
    int nintervals=n-1;
    
    //printf("%i minima, %i maxima, %i intervals\n", n, nmax, nintervals);    
    
//    printf("Minima:\n");
//    for (int k=0; k < n; k++) printf("%i %i\n", k, separators[k]);
//    printf("Maxima:\n");
//    for (int k=0; k < nmax; k++) printf("%i %i\n", k, maxima[k]);
    

    std::vector<costdata> listcosts;
    
    //Try merging intervals 1+j consecutive intervals (current + j consecutive)
    int j=1;
   
    while (nintervals > j) {
    
        //merging flag: initially set to 1
        unsigned char do_merging=1;
        while (do_merging && (nintervals > j)) {        
                     
            //compute cost of merging each interval with its successive interval
            //and find lowest cost
            struct costdata cdata, cdatalowest;
            int ilowest=-1;
            for (int i=0; i < nintervals-j; i++) {
            
                //in case of circular histogram do not try merging if the interval
                //is larger than the original input histogram
                if (circularhist && (separators[i+j+1]-separators[i] > sizeIn)) continue;
                
                if (!cost_already_computed(listcosts, separators[i], separators[i+j+1], cdata)) {
                    cdata=cost_merging(hist, listcosts, separators, maxima, i, i+j+1, logeps);  
                    //printf("    Cost merging %i-%i: %2.2f\n", separators[i], separators[i+j+1], cdata.cost);             
                }
                if ((ilowest == -1) || (cdata.cost < cdatalowest.cost)) {
                    cdatalowest=cdata;
                    ilowest=i;
                }
            }
            
            //printf("Merge %i-%i   Cost=%2.2f\n", cdatalowest.imin1, cdatalowest.imin2, cdatalowest.cost); 
            if (cdatalowest.cost > 0)  printf("Merge %i-%i   Cost=%2.2f\n", cdatalowest.imin1, cdatalowest.imin2, cdatalowest.cost);      
            //merge intervals with lowest cost, if it is smaller than 0
            if ((ilowest != -1) && (cdatalowest.cost < cost_thresh)) {

//                printf("Merge %i-%i   Cost=%2.2f\n", cdatalowest.imin1, cdatalowest.imin2, cdatalowest.cost);
                //remove minima with index ilowest+1 to ilowest+j
                separators.erase(separators.begin() + ilowest+1, separators.begin() + ilowest+j+1);                
                //remove maxima associated to the removed minima
                if (cdatalowest.typemerging == 1) {
                    maxima.erase(maxima.begin() + ilowest, maxima.begin() + ilowest+j);
                }
                if (cdatalowest.typemerging == 2) {
                    maxima.erase(maxima.begin() + ilowest+1, maxima.begin() + ilowest+j+1);
                }
                n=separators.size();
                nintervals=n-1;
            } else do_merging=0; //stop merging
            
        }
    
        j++; //increase number of intervals to merge
    }
   
    if (circularhist) {
        //output results for central copy
        std::vector<int> separatorsC;
        for (int k=0; k < separators.size(); k++) {
            if ((separators[k] >= sizeIn) && (separators[k] < 2*sizeIn)) {
                separatorsC.push_back(separators[k]-sizeIn);
            }
        }
        //if no separators then add endpoints
        if (separatorsC.size() == 0) {
            separatorsC.push_back(0);
            separatorsC.push_back(sizeIn-1);                    
        }
        delete[] hist;
        return separatorsC;
    } 
    
    return separators;
}

std::vector<int> FTCsegmentation(float *histIn, int sizeIn, float eps, 
                            unsigned char circularhist, std::vector<int>& separators)
{   
    float *hist;
    int size;
    
    double logeps=log(eps); //eps=NFAmax=maximum expected number of false alarms
    
    if (circularhist) {
        //if circular histogram then triplicate it, compute separators, and
        //output results for central copy
        size=3*sizeIn;
        hist=new float[size];
        for (int i=0; i < sizeIn; i++) {
            hist[i]=histIn[i];
            hist[i+sizeIn]=histIn[i];
            hist[i+2*sizeIn]=histIn[i];
        }
    } else {
        hist=histIn;
        size=sizeIn;
    }
           
    std::vector<int> maxima;
    maxima=get_maxima(hist, size);
    
    //check maxima
    int nmax=maxima.size();    
    
    //number of initial separators
    int n=separators.size();
    //number of intervals
    int nintervals=n-1;
    
    //printf("%i minima, %i maxima, %i intervals\n", n, nmax, nintervals);    
    
//    printf("Minima:\n");
//    for (int k=0; k < n; k++) printf("%i %i\n", k, separators[k]);
//    printf("Maxima:\n");
//    for (int k=0; k < nmax; k++) printf("%i %i\n", k, maxima[k]);
    

    std::vector<costdata> listcosts;
    
    //Try merging intervals 1+j consecutive intervals (current + j consecutive)
    int j=1;
   
    while (nintervals > j) {
    
        //merging flag: initially set to 1
        unsigned char do_merging=1;
        while (do_merging && (nintervals > j)) {        
                     
            //compute cost of merging each interval with its successive interval
            //and find lowest cost
            struct costdata cdata, cdatalowest;
            int ilowest=-1;
            for (int i=0; i < nintervals-j; i++) {
            
                //in case of circular histogram do not try merging if the interval
                //is larger than the original input histogram
                if (circularhist && (separators[i+j+1]-separators[i] > sizeIn)) continue;
                
                if (!cost_already_computed(listcosts, separators[i], separators[i+j+1], cdata)) {
                    cdata=cost_merging(hist, listcosts, separators, maxima, i, i+j+1, logeps);  
                    //printf("    Cost merging %i-%i: %2.2f\n", separators[i], separators[i+j+1], cdata.cost);             
                }
                if ((ilowest == -1) || (cdata.cost < cdatalowest.cost)) {
                    cdatalowest=cdata;
                    ilowest=i;
                }
            }
                        
            //merge intervals with lowest cost, if it is smaller than 0
            if ((ilowest != -1) && (cdatalowest.cost < 0)) {
//                printf("Merge %i-%i   Cost=%2.2f\n", cdatalowest.imin1, cdatalowest.imin2, cdatalowest.cost);
                //remove minima with index ilowest+1 to ilowest+j
                separators.erase(separators.begin() + ilowest+1, separators.begin() + ilowest+j+1);                
                //remove maxima associated to the removed minima
                if (cdatalowest.typemerging == 1) {
                    maxima.erase(maxima.begin() + ilowest, maxima.begin() + ilowest+j);
                }
                if (cdatalowest.typemerging == 2) {
                    maxima.erase(maxima.begin() + ilowest+1, maxima.begin() + ilowest+j+1);
                }
                n=separators.size();
                nintervals=n-1;
            } else do_merging=0; //stop merging
            
        }
    
        j++; //increase number of intervals to merge
    }
   
    if (circularhist) {
        //output results for central copy
        std::vector<int> separatorsC;
        for (int k=0; k < separators.size(); k++) {
            if ((separators[k] >= sizeIn) && (separators[k] < 2*sizeIn)) {
                separatorsC.push_back(separators[k]-sizeIn);
            }
        }
        //if no separators then add endpoints
        if (separatorsC.size() == 0) {
            separatorsC.push_back(0);
            separatorsC.push_back(sizeIn-1);                    
        }
        delete[] hist;
        return separatorsC;
    } 
    
    return separators;
}



