
package hw4.boosting.general;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
public class ECOC {

	static int len=18846;
	static int featlen=1754;
	static int classno=8;
	static int trainlen=11314,testlen=7532;
	static int codesize=127;
	static int stop=1000;
	static double notpresent=-99.0;
	static double [] labelstrain = new double[trainlen];
	static double [] labelstest= new double[testlen];
	static double [] codelabelstrain = new double[trainlen];
	static double [] codelabelstest= new double[testlen];
	static double [] predictiontest;
	static double [] predictiontrain;
	static double [] epredictiontest;
	static double [] epredictiontrain;
	static double[][] totalpredictiontest = new double[codesize][testlen];
	static double[][] totalpredictiontrain = new double[codesize][trainlen];
	static double[][] etotalpredictiontest = new double[codesize][testlen];
	static double[][] etotalpredictiontrain = new double[codesize][trainlen];
	static double dataweights[];
	static ArrayList<HashMap<Integer,Double>> featurestrain = new ArrayList<HashMap<Integer,Double>>();
	static ArrayList<HashMap<Integer,Double>> featurestest = new ArrayList<HashMap<Integer,Double>>();
	static HashMap<Integer,TreeSet<Double>> featcollection = new HashMap<Integer,TreeSet<Double>>();
	static ArrayList<double[]> codelabels = new ArrayList<double[]>();
	static ArrayList<ArrayList<Double>> stumps=new ArrayList<ArrayList<Double>>();
	static ArrayList<Double> alphat;
	static ArrayList<Integer> stumpt;
	static ArrayList<Double> threstt;
	static ArrayList<Double> errort;
	static ArrayList<Double> auct;
	static ArrayList<Double> trainerrort;
	static ArrayList<Double> testerrort;
	static int runs;
	static int totalstump;
	private static void readData()
	{
		try {
			FileReader featureread = new FileReader("data/8newsgroup/train.trec/feature_matrix.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				labelstrain[ind]=Double.parseDouble(feats[0]);
				HashMap<Integer,Double> featlist = new HashMap<Integer,Double>();
				for(int i=1;i<feats.length;i++)
				{
					String[] featsplit=feats[i].split(":");
					int featno=Integer.parseInt(featsplit[0]);
					double val=Double.parseDouble(featsplit[1]);
					featlist.put(featno,val);
					if(featcollection.containsKey(featno))
					{
						TreeSet<Double> temp = featcollection.get(featno);
						temp.add(val);
						featcollection.put(featno, temp);
					}
					else
					{
						TreeSet<Double> temp = new TreeSet<Double>();
						temp.add(val);
						temp.add(notpresent);
						featcollection.put(featno, temp);
					}
				}
				featurestrain.add(ind,featlist);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/8newsgroup/test.trec/feature_matrix.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				labelstest[ind]=Double.parseDouble(feats[0]);
				HashMap<Integer,Double> featlist = new HashMap<Integer,Double>();
				for(int i=1;i<feats.length;i++)
				{
					String[] featsplit=feats[i].split(":");
					featlist.put(Integer.parseInt(featsplit[0]),Double.parseDouble(featsplit[1]));
				}
				featurestest.add(ind,featlist);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	private static void setClassCodes(boolean write,boolean read){
		if(read)
		{
			try {
				FileReader featureread = new FileReader("data/codes");
				BufferedReader featurereadbr = new BufferedReader(featureread);
				String sCurrentLine;
				String[] feats;
				int ind=0;
				while ((sCurrentLine = featurereadbr.readLine()) != null) {
					feats=sCurrentLine.split(" ");
					double code[]=new double[codesize];
					for(int i=0;i<feats.length;i++)
					{
						code[i]=Double.parseDouble(feats[i]);	
					}
					codelabels.add(ind,code);
					ind++;
				}
				featurereadbr.close();
				featureread.close();
			}catch(Exception e){
				e.printStackTrace();
			}
			return;
		}
		Random n= new Random();
		HashMap<Integer, Boolean> code= new HashMap<Integer,Boolean>();
		int val=0;
		for (int i = 0; i < classno; i++) {
			double[] c=new double[codesize];
			for (int j = 0; j < c.length; j++) {
				c[j]=0;
			}
			val=0;int s=0;HashMap<Integer, Boolean> roundval= new HashMap<Integer,Boolean>();
			while(true){
				int b=-1;

				b=n.nextInt(codesize);
				if(!roundval.containsKey(b))
					val+=Math.pow(2, b);
				roundval.put(b,false );
				c[b]=1;
				//System.out.println(i+"  "+s+"  "+b+" "+val);
				s++;
				if(s>codesize-1)
				{
					if(!code.containsKey(val))
						break;
					else
					{
						s=0;
						for (int j = 0; j < c.length; j++) {
							c[j]=0;
						}
						val=0;
						roundval= new HashMap<Integer,Boolean>();
						continue;
					}
				}
			}
			//System.out.println(val);
			code.put(val, false);
			codelabels.add(i, c);
		}
		if(write){
			try {
				FileWriter fwc = new FileWriter("data/codes");
				BufferedWriter bwc = new BufferedWriter(fwc);
				for (int i = 0; i < codelabels.size(); i++) {
					for (int j = 0; j < codesize; j++) {
						bwc.write(codelabels.get(i)[j]+" ");
					}
					bwc.newLine();
				}
				bwc.close();
				fwc.close();

			}catch (Exception e){
				System.out.println(e.getMessage());	
			}
		}
	}
	private static void setHammingCodes(){
		for (int i = 1; i <= classno; i++) {
			double[] c=new double[codesize];
			int block=(int) Math.pow(2, classno-i);
			int set=1,cnt=0;
			for (int j = 0; j < c.length; j++) {
				cnt++;
				if(cnt==block)
				{
					if(set==1)
						set=0;
					else
						set=1;
					cnt=0;
				}
				c[j]=set;
			}
			codelabels.add(i-1, c);
		}
	}
	private static void setCodes(){
		for (int i = 1; i <= classno; i++) {
			double[] c=new double[codesize];
			for (int j = 1; j <= c.length; j++) {
				if((j*j+i*i)%2==0)
					c[j-1]=1;
			}
			codelabels.add(i-1, c);
		}
	}
	private static void setDataweights(){
		dataweights=new double[trainlen] ;
		for (int i = 0; i < trainlen; i++) {
			dataweights[i]=1.0/(double) trainlen;
		}
	}
	private static void createStumps(){
		for (int i = 0; i < featlen; i++) {
			ArrayList<Double> featstump=new ArrayList<Double>();
			TreeSet<Double> enefeat = featcollection.get(i);
			boolean first=true;
			double preval=0.0;
			double thresh=0.0;
			double avgthresh=0.0;
			for (Double tval : enefeat) {
				if(first)
				{
					preval=tval;
					first=false;
					continue;
				}
				thresh=(tval-preval)/2.0;
				preval=tval;
				avgthresh+=thresh;
				featstump.add(thresh);
			}
			avgthresh/=(double) (enefeat.size()-1);
			featstump.add(0,enefeat.last()+avgthresh);
			featstump.add(enefeat.first()-avgthresh);
			totalstump+=featstump.size();
			stumps.add(i, featstump);
		}
	}
	private static void init(int codenum){
		stumps=new ArrayList<ArrayList<Double>>();
		alphat=new ArrayList<Double>();
		stumpt=new ArrayList<Integer>();
		threstt=new ArrayList<Double>();
		errort=new ArrayList<Double>();
		auct=new ArrayList<Double>();
		trainerrort=new ArrayList<Double>();
		testerrort=new ArrayList<Double>();
		runs =0;
		totalstump= 0;
		setDataweights();
		createStumps();
		for (int i = 0; i < testlen; i++) {
			codelabelstest[i]=codelabels.get((int) labelstest[i])[codenum];
		}
		for (int i = 0; i < trainlen; i++) {
			codelabelstrain[i]=codelabels.get((int) labelstrain[i])[codenum];
		}
		predictiontest= new double[testlen];
		predictiontrain= new double[testlen];
		epredictiontest= new double[testlen];
		epredictiontrain= new double[testlen];
	}
	private static double getStumperror(int feat,double thresh){
		double error=0.0;
		for (int i = 0; i < trainlen; i++) {
			double val=0.0;
			if(featurestrain.get(i).containsKey(feat))
				val=featurestrain.get(i).get(feat);
			if(val>thresh)
			{
				if(codelabelstrain[i]==0)
				{
					error+=dataweights[i];
				}
			}
			else
			{
				if(codelabelstrain[i]==1)
				{
					error+=dataweights[i];
				}
			}
		}
		return error;
	}
	private static double[] getRandStump(){
		Random n = new Random();
		int stumpno=n.nextInt(stumps.size());
		int threshno=0;
		while(true){
			stumpno=n.nextInt(stumps.size());
			if(stumps.get(stumpno).size()!=0)
			{ 
				threshno=n.nextInt(stumps.get(stumpno).size());
				break;
			}
		}
		double thresh=stumps.get(stumpno).get(threshno);
		double reverror=getStumperror(stumpno, thresh);
		ArrayList<Double> temp = stumps.get(stumpno);
		temp.remove(thresh);
		stumps.set(stumpno, temp);
		double ans[]={stumpno,thresh,reverror};
		return ans;
	}
	private static double[] getBestStump(){
		double error=0.0,temperror=0.0,reverror=0.0;
		int stumpno=0;
		double thresh=0.0;
		for (int i = 0; i < featlen; i++) {
			ArrayList<Double> featsump = stumps.get(i);
			for (int j = 0; j < featsump.size(); j++) {
				temperror=getStumperror(i, featsump.get(j));
				if(Math.abs(0.5-temperror)>error)
				{
					reverror=temperror;
					error=Math.abs(0.5-temperror);
					stumpno=i;
					thresh=featsump.get(j);
				}
			}
		}
		ArrayList<Double> temp = stumps.get(stumpno);
		temp.remove(thresh);
		stumps.set(stumpno, temp);
		double ans[]={stumpno,thresh,reverror};
		return ans;
	}
	private static void oneRun(boolean mode){
		double stump[]=new double[3];
		if(mode)
			stump=getBestStump();
		else
			stump=getRandStump();
		double alpha=0.5*Math.log((1.0-stump[2])/(stump[2]));
		alphat.add(alpha);
		stumpt.add((int) stump[0]);
		threstt.add(stump[1]);
		errort.add(stump[2]);
		double zt=0.0;
		for (int i = 0; i < trainlen; i++) {
			double val=notpresent;
			if(featurestrain.get(i).containsKey((int) stump[0]))
				val=featurestrain.get(i).get((int) stump[0]);
			if(val>stump[1])
			{
				if(codelabelstrain[i]==0)
				{
					dataweights[i]*=Math.exp(alpha);
				}
				else
				{
					dataweights[i]*=Math.exp(-alpha);
				}
			}
			else
			{
				if(codelabelstrain[i]==1)
				{
					dataweights[i]*=Math.exp(alpha);
				}
				else
				{
					dataweights[i]*=Math.exp(-alpha);
				}
			}
			zt+=dataweights[i];
		}
		for (int i = 0; i < trainlen; i++) {
			dataweights[i]/=zt;
		}
		runs++;
	}
	private static double[] prediction(ArrayList<HashMap<Integer,Double>> featureseval,double[] labelseval,int tetrlen){
		double[] predicted=new double[tetrlen];
		for (int i = 0; i < tetrlen; i++) {
			predicted[i]=0.0;
			for (int j = 0; j < runs; j++) {
				double val=notpresent;
				if(featureseval.get(i).containsKey(stumpt.get(j)))
					val=featureseval.get(i).get(stumpt.get(j));
				if(val>threstt.get(j))
				{
					predicted[i]+=alphat.get(j);
				}
			}
		}
		return predicted;
	}
	private static void getResult(ArrayList<HashMap<Integer,Double>> featureseval,double[] labelseval,int tetrlen,boolean auc,boolean write){
		double prediction[]=prediction(featureseval, labelseval, tetrlen);
		TreeMap<Double, Integer[]> allpredict=new TreeMap<Double, Integer[]>();
		double error=tetrlen;
		int totalone=0;
		for (int i = 0; i < tetrlen; i++) {
			Integer[] oz={0,0};
			if(allpredict.containsKey(prediction[i]))
				oz=allpredict.get(prediction[i]);
			if(labelseval[i]==0)
				oz[0]++;
			else
				oz[1]++;
			allpredict.put(prediction[i], oz);
			totalone+=labelseval[i];
		}
		int iter=0,tp=totalone,fp=tetrlen-totalone,tn=0,fn=0;
		double thresh=0.0,prevthresh=0.0;
		ArrayList<Integer[]> ccall=new ArrayList<Integer[]>();
		double itererror=0.0;
		itererror=fp+fn;
		itererror/=(double) tetrlen;
		if(itererror<error)
		{
			error=itererror;
			thresh=allpredict.firstKey()-0.01;
		}
		for(Double key:allpredict.keySet()){
			iter++;
			Integer[] oz = allpredict.get(key);
			tp-=oz[1];
			fn+=oz[1];
			fp-=oz[0];
			tn+=oz[0];
			itererror=fp+fn;
			itererror/=(double) tetrlen;
			if(itererror<error)
			{
				//System.out.println("inga");
				error=itererror;
				thresh=(key+prevthresh)/2.0;
			}
			prevthresh=key;
		}
		iter++;tp=0;fp=0;tn=tetrlen-totalone;fn=totalone;
		itererror=fp+fn;
		itererror/=(double) tetrlen;
		if(itererror<error)
		{
			//System.out.println("enga");
			error=itererror;
			thresh=prevthresh;
		}
		if(auc)
		{
			//y2=((double)tp)/((double)tp+fn);
			//x2=((double)fp)/((double)fp+tn);
			//aucval+=(0.5*(y1+y2)*(x1-x2));
			testerrort.add(error);
			//auct.add(aucval);
		}
		if(!auc)
		{
			trainerrort.add(error);
		}
		double[] common= new double[tetrlen];
		for (int i = 0; i < tetrlen; i++) {
			//System.out.println(prediction[i]+"  "+thresh);
			if(prediction[i]>thresh)
				common[i]=1;
			else
				common[i]=0;
		}
		/*error=0;
		for (int i = 0; i < tetrlen; i++) {
			if(common[i]!=labelseval[i])
				error++;
		}
		//error/=(double) tetrlen;
		System.out.println(error+" "+f);*/
		double[] predictionerror= new double[tetrlen];
		for (int i = 0; i < tetrlen; i++) {
			if(common[i]!=labelseval[i])
				predictionerror[i]=1;
			else
				predictionerror[i]=0;
		}
		if(tetrlen==testlen)
		{
			predictiontest=common;
			epredictiontest=predictionerror;
		}
		else
		{
			predictiontrain=common;
			epredictiontrain=predictionerror;
		}

	}
	private static void allput(int codeno,boolean mode,boolean graph){
		init(codeno);
		long startfold = System.currentTimeMillis( );
		boolean converged=false;
		while(!converged){
			oneRun(mode);
			/*getResult(featurestrain, codelabelstrain, trainlen, false,false);
			getResult(featurestest, codelabelstest, testlen, true,false);
			System.out.println(trainerrort.get(runs-1));
			*/ if(runs>stop)
				 converged=true;
			// System.out.println("Iteration No: "+(runs)+" Feature No: "+(stumpt.get(runs-1)+1)+" Threshold: "+threstt.get(runs-1)+" Rounderror: "+errort.get(runs-1)+" Trainerror: "+trainerrort.get(runs-1)+" Testerror: "+testerrort.get(runs-1));
		}
		getResult(featurestrain, codelabelstrain, trainlen, false,false);
		getResult(featurestest, codelabelstest, testlen, true,false);
		System.out.println("Iteration No: "+(runs)+" Feature No: "+(stumpt.get(runs-1)+1)+" Threshold: "+threstt.get(runs-1)+" Rounderror: "+errort.get(runs-1)+" Trainerror: "+trainerrort.get(1-1)+" Testerror: "+testerrort.get(1-1));
		long endfold = System.currentTimeMillis( );
		System.out.println("Process took "+(endfold-startfold)/1000+" secs");
	}
	private static double[] convert(double[][] totalpredict,double[][] etotalpredict,int tetrlen){
		double[] finalpredict=new double[tetrlen];
		for (int i = 0; i < tetrlen; i++) {
			//System.out.println(i+" "+labelstrain[i]);
			double[] onedp=new double[codesize];
			int diff=0,clas=0,maxdiff=300;
			for (int j = 0; j < codesize; j++) {
				onedp[j]=totalpredict[j][i];
			}
			for (int j = 0; j < classno; j++) {
				diff=0;
				for (int k = 0; k < codesize; k++) {
					if(onedp[k]!=codelabels.get(j)[k])
						diff++;
				}

				/*System.out.println("Wrong Ind:"+j);
				for (int k = 0; k < codesize; k++) {
					System.out.print(etotalpredict[k][i]+" ");
				}
				System.out.println();
				System.out.println("Class No:"+j);
				for (int k = 0; k < codesize; k++) {
					System.out.print(onedp[k]+" ");
				}
				System.out.println();
				for (int k = 0; k < codesize; k++) {
					System.out.print(codelabels.get(j)[k]+" ");
				}
				System.out.println(diff+"  "+maxdiff);*/
				if(diff<maxdiff)
				{
					maxdiff=diff;
					clas=j;
				}
			}
			/*try {
				System.in.read();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}*/
			//System.out.println(clas);
			finalpredict[i]=clas;
		}
		return finalpredict;
	}
	private static double[] getAccuracy(double[] predictions,double[] labelseval,int tetrlen){
		double acc=0;
		for (int i = 0; i < tetrlen; i++) {
			//System.out.println(predictions[i]+"  "+labelseval[i]);
			if(predictions[i]!=labelseval[i])
				acc++;
		}
		double[] ans={acc,(tetrlen-acc)/(double) tetrlen};
		return ans;
	}
	public static void main(String[] args) {
		readData();
		setHammingCodes();
		//setClassCodes(false,false);
		//setCodes();
		for (int i = 0; i < codesize; i++) {
			System.out.println(i);
			allput(i, false, false);
			totalpredictiontest[i]=predictiontest;
			totalpredictiontrain[i]=predictiontrain;
			etotalpredictiontest[i]=epredictiontest;
			etotalpredictiontrain[i]=epredictiontrain;
		}

		/*for (int i = 0; i < testlen; i++) {
			for (int j = 0; j < codesize; j++) {
				System.out.print(totalpredictiontest[j][i]+"  ");
			}
			System.out.println();
		}*/
		/*for (int i = 0; i < trainlen; i++) {
			int sum=0;
			for (int j = 0; j < codesize; j++) {
				sum+= etotalpredictiontrain[j][i];
				System.out.print(etotalpredictiontrain[j][i]+" ");
			}
			System.out.println(" "+sum+"  "+labelstrain[i]);
		}*/
		double[] traintp = convert(totalpredictiontrain,etotalpredictiontrain, trainlen);
		double[] testp = convert(totalpredictiontest,etotalpredictiontest, testlen);
		double[] acctest = getAccuracy(testp, labelstest, testlen);
		double[] acctrain = getAccuracy(traintp, labelstrain, trainlen);
		System.out.println("Training Accuracy: "+acctrain[1]);
		System.out.println("Testing Accuracy: "+acctest[1]);

		/*for (int i = 0; i < classno; i++) {
			for (int j = 0; j < codesize; j++) {
				System.out.print(codelabels.get(i)[j]+" ");
			}System.out.println();
		}*/
	}

}
