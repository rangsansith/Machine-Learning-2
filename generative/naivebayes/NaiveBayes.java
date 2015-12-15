package hw3.generative.naivebayes;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class NaiveBayes {
	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int trainlen,testlen;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static ArrayList<double[][]> featklist = new ArrayList<double[][]>();
	static ArrayList<double[]> labelsklist = new ArrayList<double[]>();
	static ArrayList<Integer> indexlist = new ArrayList<Integer>();
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [] labelstest= new double[1];
	static Double[] globalmean;
	private static void readData()
	{
		int one=0,zero=0;
		try {
			FileReader featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				indexlist.add(ind);
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);

				}
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labels[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
	}
	private static void randSplits(int folds)
	{
		HashMap<Integer, Boolean> seen = new HashMap<Integer, Boolean>();
		int kfold=len/folds,listno=0,one=0,zero=0;
		Random n = new Random(); 
		while(indexlist.size()!=0){
			one=0;zero=0;
			int r=Math.min(kfold, indexlist.size());
			double[][] featurestemp = new double[r][featlen];
			double [] labelstemp = new double[r];
			for (int i = 0; i < kfold; i++) {
				if(indexlist.size()==0)
					break;
				int ind=n.nextInt(indexlist.size());
				int tmpind=indexlist.get(ind);
				if(seen.containsKey(tmpind))
					System.out.println("Error: bad Coder");
				else
					seen.put(tmpind, true);
				for (int j = 0; j < featlen; j++) {
					featurestemp[i][j]=features[tmpind][j];
				}
				labelstemp[i]=labels[tmpind];
				if(labels[tmpind]==0.0)
					zero++;
				else if(labels[tmpind]==1.0)
					one++;
				else
					System.out.println("Error: bad Coder");

				indexlist.remove(ind);
			}
			System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void uniformSplits(int folds){
		ArrayList<ArrayList<double[]>> tft= new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> tlt= new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < folds; i++) {
			tft.add(new ArrayList<double[]>());
			tlt.add(new ArrayList<Double>());
		}
		for (int i = 0; i < len; i++) {
			ArrayList<double[]> temp = tft.get(i%folds);
			temp.add(features[i]);
			tft.set(i%folds, temp);
			ArrayList<Double> lemp = tlt.get(i%folds);
			lemp.add(labels[i]);
			tlt.set(i%folds,lemp);
		}
		if(tft.size()!=tlt.size())
			System.out.println("Error: BadCode");
		int one=0,zero=0,listno=0;
		for (int i = 0; i < tft.size(); i++) {
			one=0;zero=0;
			double[][] featurestemp = new double[tft.get(i).size()][featlen];
			double [] labelstemp = new double[tlt.get(i).size()];
			for (int j = 0; j < tft.get(i).size(); j++) {
				featurestemp[j]=tft.get(i).get(j);
				labelstemp[j]=tlt.get(i).get(j);
				if(labelstemp[j]==0)
					zero++;
				else
					one++;
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void pickKfold(int k){
		k--;
		testlen=labelsklist.get(k).length;
		trainlen=len-testlen;
		featurestrain=new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest=new double[testlen][featlen];
		labelstest=new double[testlen];
		int ind=0;
		for (int i = 0; i < featklist.size(); i++) {
			if(i==k)
			{
				featurestest=featklist.get(k);
				labelstest=labelsklist.get(k);
				continue;
			}
			double[][] tf = featklist.get(i);
			double[] tl = labelsklist.get(i);
			for (int j = 0; j < tl.length; j++) {
				featurestrain[ind]=tf[j];
				labelstrain[ind]=tl[j];
				ind++;
			}
		}
	}
	private static Double[] getMean(double[][] features,int x,int y){
		Double[] mean=new Double[y];
		for (int i = 0; i < y; i++) {
			mean[i]=0.0;
			for (int j = 0; j < x; j++) {
				mean[i]+=features[j][i];
			}
			mean[i]/=(double) x;
		}
		return mean;
	}
	private static Double[][] getMinMax(double[][] features,int x,int y){
		Double[][] minmax=new Double[2][y];
		for (int i = 0; i < y; i++) {
			minmax[0][i]=1000000000000.0;
			minmax[1][i]=-100000000000.0;
			for (int j = 0; j < x; j++) {
				minmax[0][i]=Math.min(features[j][i],minmax[0][i]);
				minmax[1][i]=Math.max(features[j][i],minmax[1][i]);
			}
		}
		return minmax;
	}
	private static Double[] getStd(double[][] features,int x,int y,Double[] mean){
		Double[] std=new Double[y];
		for (int i = 0; i < y; i++) {
			std[i]=0.0;
			for (int j = 0; j < x; j++) {
				std[i]+=Math.pow(features[j][i]-mean[i],2);
			}
			std[i]=Math.sqrt(std[i]/(double) x);
		}
		return std;
	}
	private static Double[] getVar(double[][] features,int x,int y,Double[] mean){
		Double[] std=new Double[y];
		for (int i = 0; i < y; i++) {
			std[i]=0.0;
			for (int j = 0; j < x; j++) {
				std[i]+=Math.pow(features[j][i]-mean[i],2);
			}
			std[i]=(std[i]/x);
		}
		return std;
	}
	private static Double[][][] getMinMaxbyClass(double[][] features,double[] labels,int x,int y,double[] val){
		Double[][][] minmax=new Double[val.length][2][y];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				minmax[k][0][i]=1000000000000.0;
				minmax[k][1][i]=-100000000000.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						minmax[k][0][i]=Math.min(features[j][i],minmax[k][0][i]);
						minmax[k][1][i]=Math.max(features[j][i],minmax[k][1][i]);
					}
				}
			}
		}
		return minmax;
	}
	private static Double[][] getMeanbyClass(double[][] features,double[] labels,int x,int y,double[] val){
		Double[][] mean=new Double[val.length][y];
		Double[] len=new Double[val.length];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				mean[k][i]=0.0;
				len[k]=0.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						mean[k][i]+=features[j][i];
						len[k]++;
					}
				}
				mean[k][i]/=len[k];
			}
		}
		return mean;
	}
	private static Double[][] getCommonVarbyClass(double[][] features,double[] labels,int x,int y,double[] val, Double[][] mean){
		Double[][] std=new Double[val.length][y];
		for (int i = 0; i < y; i++) {
			std[0][i]=0.0;
			for (int j = 0; j < x; j++) 
			{
				std[0][i]+=Math.pow(features[j][i]-mean[(int) labels[j]][i],2);
			}
			std[0][i]/=(double) x;
		}
		std[1]=std[0];
		return std;
	}
	private static Double[][] getBiasedVarbyClass(double[][] features,double[] labels,int x,int y,double[] val,Double[][] mean){
		Double[][] std=new Double[val.length][y];
		Double[] len=new Double[val.length];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				std[k][i]=0.0;
				len[k]=0.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						std[k][i]+=Math.pow(features[j][i],2);
						len[k]++;
					}
				}
				std[k][i]/=(len[k])-mean[k][i];
			}
		}
		return std;
	}
	private static Double[][] getVarbyClass(double[][] features,double[] labels,int x,int y,double[] val,Double[][] mean){
		Double[][] std=new Double[val.length][y];
		Double[] len=new Double[val.length];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				std[k][i]=0.0;
				len[k]=0.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						std[k][i]+=Math.pow(features[j][i]-mean[k][i],2);
						len[k]++;
					}
				}
				std[k][i]/=(len[k]-1);
			}
		}
		return std;
	}
	private static void normalize(double[][] features,int x,int y,Double[] mean,Double[] std){
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				features[j][i]-=mean[i];
				features[j][i]/=std[i];
			}
		}
	}
	private static void noramalizeboth(double[][] features,double[][] featurestest,int trx,int tex,int y){
		Double[] mean=getMean(features, trx, y);
		Double[] std=getStd(features, trx, y, mean);
		normalize(features, trx, y, mean, std);
		normalize(featurestest, tex, y, mean, std);
	}
	private static double[][] ROCBernoulli(double[][]featureseval,double[] labelseval,double[][][] conditionalprob,double spam,double ham,int tetrlen)
	{
		ArrayList<Double> threshlist = new ArrayList<Double>();
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=conditionalprob[k][(featureseval[j][k]>globalmean[k]?1:0)][1];
				hamprob*=conditionalprob[k][(featureseval[j][k]>globalmean[k]?1:0)][0];
			}
			threshlist.add(Math.log(spamprob/hamprob));
		}
		Collections.sort(threshlist);
		threshlist.add(0,threshlist.get(0)-1);
		threshlist.add(threshlist.size(),threshlist.get(threshlist.size()-1)+1);
		double[][] thrcc=new double[threshlist.size()][4];
		for (int i = 0; i < threshlist.size(); i++) {
			evaluateBernoulli(featureseval, labelseval, conditionalprob, thrcc[i], spam, ham, tetrlen, threshlist.get(i));
		}
		return thrcc;
	}
	private static double evaluateBernoulli(double[][]featureseval,double[] labelseval,double[][][] conditionalprob,double cc[],double spam,double ham,int tetrlen,double threshold)
	{
		double acc=0;
		double tp=0.0,fp=0.0,tn=0.0,fn=0.0;
		// Testing the model
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=conditionalprob[k][(featureseval[j][k]>globalmean[k]?1:0)][1];
				hamprob*=conditionalprob[k][(featureseval[j][k]>globalmean[k]?1:0)][0];
			}
			if(Math.log(spamprob/hamprob)>threshold)
			{
				if(labelseval[j]==1)
				{
					acc++;
					tp++;
				}
				else
				{
					fp++;
				}
			}
			else
			{
				if(labelseval[j]==0)
				{
					acc++;
					tn++;
				}
				else
				{
					fn++;
				}
			}
		}
		acc/=((double) tetrlen);
		cc[0]=tp;
		cc[1]=fp;
		cc[2]=tn;
		cc[3]=fn;
		//System.out.println("Accuracy: "+acc*100+"  "+spam+"  "+ham+"  "+tp+"  "+fp+"  "+tn+"  "+fn);
		return acc;
	}
	private static double[][] Bernoulli(double[][] cc,double[] acc,boolean mode){
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		// Estimate conditional probability
		int category=2,spamcount=0;
		double[][][] conditionalprob= new double[featlen][category][classno];
		for (int j = 0; j < featlen; j++) {
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]=1;
				}
			}
			spamcount=0;
			for (int k = 0; k < trainlen; k++) {
				conditionalprob[j][(featurestrain[k][j]>globalmean[j]?1:0)][(int) labelstrain[k]]++;
				spamcount+=labelstrain[k];
			}
			int classlen[]={trainlen-spamcount+category,spamcount+category};
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]/=((double) (classlen[m]));//+classno*category));
				}
			}
		}
		double spam=((double) spamcount)/((double) trainlen);
		double ham=1.0-spam;
		//evaluate
		if(!mode)
		{
			acc[0]=evaluateBernoulli(featurestrain, labelstrain, conditionalprob, cc[0], spam, ham, trainlen,0.0);
			acc[1]=evaluateBernoulli(featurestest, labelstest, conditionalprob, cc[1], spam, ham, testlen,0.0);
		}
		else
		{
			return ROCBernoulli(featurestest, labelstest, conditionalprob, spam, ham, testlen);
		}
		return null;
	}
	private static double GaussianModel(double mu,double var,double actval){
		double ans=1.0/(Math.sqrt(2.0*Math.PI*var));
		ans*=Math.exp(-Math.pow((actval-mu), 2)/(2.0*var));
		return ans;
	}
	private static double[][] ROCGaussian(double[][]featureseval,double[] labelseval,Double[][] classmean,Double[][] classvar,double spam,double ham,int tetrlen)
	{
		ArrayList<Double> threshlist = new ArrayList<Double>();
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=GaussianModel(classmean[1][k], classvar[1][k], featureseval[j][k]);
				hamprob*=GaussianModel(classmean[0][k], classvar[0][k], featureseval[j][k]);
			}
			if(spamprob==0)
			{
				spamprob=hamprob/10;
			}
			if(hamprob==0)
			{
				hamprob=spamprob/10;
			}
			threshlist.add(Math.log(spamprob/hamprob));
		}
		Collections.sort(threshlist);
		threshlist.add(0,threshlist.get(0)-1);
		threshlist.add(threshlist.size(),threshlist.get(threshlist.size()-1)+1);
		double[][] thrcc=new double[threshlist.size()][4];
		for (int i = 0; i < threshlist.size(); i++) {
			evaluateGaussian(featureseval, labelseval, classmean, classvar, thrcc[i], spam, ham, tetrlen,threshlist.get(i));
		}
		return thrcc;
	}
	private static double evaluateGaussian(double[][]featureseval,double[] labelseval,Double[][] classmean,Double[][] classvar,double cc[],double spam,double ham,int tetrlen,double threshold)
	{
		double acc=0;
		double tp=0.0,fp=0.0,tn=0.0,fn=0.0;
		// Testing the model
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=GaussianModel(classmean[1][k], classvar[1][k], featureseval[j][k]);
				hamprob*=GaussianModel(classmean[0][k], classvar[0][k], featureseval[j][k]);
			}
			if(spamprob==0)
			{
				spamprob=hamprob/10;
			}
			if(hamprob==0)
			{
				hamprob=spamprob/10;
			}
			if(Math.log(spamprob/hamprob)>threshold)
			{
				if(labelseval[j]==1)
				{
					acc++;
					tp++;
				}
				else
				{
					fp++;
				}
			}
			else
			{
				if(labelseval[j]==0)
				{
					acc++;
					tn++;
				}
				else
				{
					fn++;
				}
			}
		}
		acc/=((double) tetrlen);
		cc[0]=tp;
		cc[1]=fp;
		cc[2]=tn;
		cc[3]=fn;
		//System.out.println("Accuracy: "+acc*100+"  "+spam+"  "+ham+"  "+tp+"  "+fp+"  "+tn+"  "+fn);
		return acc;
	}
	private static double[][] Gaussian(double[][] cc,double[] acc, boolean mode){
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		double[] val={0,1};
		Double[][] classmean=getMeanbyClass(featurestrain, labelstrain, trainlen, featlen, val);
		Double[][] classvar=getCommonVarbyClass(featurestrain, labelstrain, trainlen, featlen, val, classmean);
		int spamcount=0;
		 for (int j = 0; j < trainlen; j++) {
			 spamcount+=labelstrain[j];
		 }
		 double spam=((double) spamcount)/((double) trainlen),ham=1-spam;
		 //evaluate
		 if(!mode)
		 {
			 acc[0]=evaluateGaussian(featurestrain, labelstrain, classmean, classvar, cc[0], spam, ham, trainlen,0.0);
			 acc[1]=evaluateGaussian(featurestest, labelstest, classmean, classvar, cc[1], spam, ham, testlen,0.0);
		 }
		 else
		 {
			 return ROCGaussian(featurestest, labelstest, classmean, classvar, spam, ham, testlen);
		 }
		 return null;
	}
	private static int selectBin(double[][] tempbin,double val)
	{
		for (int i = 0; i < tempbin.length; i++) {
			if((val>=tempbin[i][0])&&(val<tempbin[i][1]))
				return i;
		}
		return -1;
	}
	private static double evaluateHist(double[][]featureseval,double[] labelseval,double[][][] conditionalprob,double[][][] bin,double cc[],double spam,double ham,int tetrlen,double threshold)
	{
		double acc=0;
		double tp=0.0,fp=0.0,tn=0.0,fn=0.0;
		// Testing the model
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=conditionalprob[k][selectBin(bin[k], featurestrain[j][k])][1];
				hamprob*=conditionalprob[k][selectBin(bin[k], featurestrain[j][k])][0];
			}
			if(Math.log(spamprob/hamprob)>threshold)
			{
				if(labelseval[j]==1)
				{
					acc++;
					tp++;
				}
				else
				{
					fp++;
				}
			}
			else
			{
				if(labelseval[j]==0)
				{
					acc++;
					tn++;
				}
				else
				{
					fn++;
				}
			}
		}
		acc/=((double) tetrlen);
		cc[0]=tp;
		cc[1]=fp;
		cc[2]=tn;
		cc[3]=fn;
		//System.out.println("Accuracy: "+acc*100+"  "+spam+"  "+ham+"  "+tp+"  "+fp+"  "+tn+"  "+fn);
		return acc;
	}
	private static double[][] ROCHist(double[][]featureseval,double[] labelseval,double[][][] conditionalprob,double[][][] bin,double spam,double ham,int tetrlen)
	{
		ArrayList<Double> threshlist = new ArrayList<Double>();
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=conditionalprob[k][selectBin(bin[k], featurestrain[j][k])][1];
				hamprob*=conditionalprob[k][selectBin(bin[k], featurestrain[j][k])][0];
			}
			threshlist.add(Math.log(spamprob/hamprob));
		}
		Collections.sort(threshlist);
		threshlist.add(0,threshlist.get(0)-1);
		threshlist.add(threshlist.size(),threshlist.get(threshlist.size()-1)+1);
		double[][] thrcc=new double[threshlist.size()][4];
		for (int i = 0; i < threshlist.size(); i++) {
			evaluateHist(featureseval, labelseval, conditionalprob, bin, thrcc[i], spam, ham, tetrlen, threshlist.get(i));
		}
		return thrcc;
	}
	private static double[][] HistFour(double[][] cc,double[] acc, boolean mode){
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		// Estimate conditional probability
		double[] val={0,1};
		double[][][] bin= new double[featlen][4][2];
		Double[] overallmean = getMean(featurestrain, trainlen, featlen);
		Double[][] classmean = getMeanbyClass(featurestrain, labelstrain, trainlen, featlen, val);
		Double[][] minmax = getMinMax(featurestrain, trainlen, featlen);
		for (int j = 0; j < featlen; j++) {
			bin[j][0][0]=minmax[0][j];
			bin[j][0][1]=classmean[0][j];
			bin[j][1][0]=classmean[0][j];
			bin[j][1][1]=overallmean[j];
			bin[j][2][0]=overallmean[j];
			bin[j][2][1]=classmean[1][j];
			bin[j][3][0]=classmean[1][j];
			bin[j][3][1]=minmax[1][j]+0.1;
		}
		int category=4,spamcount=0;
		double[][][] conditionalprob= new double[featlen][category][classno];
		for (int j = 0; j < featlen; j++) {
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]=1;
				}
			}
			spamcount=0;
			for (int k = 0; k < trainlen; k++) {
				conditionalprob[j][selectBin(bin[j], featurestrain[k][j])][(int) labelstrain[k]]++;
				spamcount+=labelstrain[k];
			}
			int classlen[]={trainlen-spamcount+category,spamcount+category};
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]/=((double) (classlen[m]));//+classno*category));
				}
			}
		}
		double spam=((double) spamcount)/((double) trainlen);
		double ham=1.0-spam;
		if(!mode)
		{
			acc[0]=evaluateHist(featurestrain, labelstrain, conditionalprob, bin, cc[0], spam, ham, trainlen,0.0);
			acc[1]=evaluateHist(featurestest, labelstest, conditionalprob, bin, cc[1], spam, ham, testlen,0.0);
		}
		else
		{
            return ROCHist(featurestest, labelstest, conditionalprob, bin, spam, ham, testlen);
		}
		return null;
	}
	private static double[][] HistNine(double[][] cc,double[] acc, boolean mode){
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		// Estimate conditional probability
		double[] val={0,1};
		double[][][] bin= new double[featlen][9][2];
		Double[] overallmean = getMean(featurestrain, trainlen, featlen);
		Double[][] classmean = getMeanbyClass(featurestrain, labelstrain, trainlen, featlen, val);
		Double[][] minmax = getMinMax(featurestrain, trainlen, featlen);
		for (int j = 0; j < featlen; j++) {
			ArrayList<Double> allval=new ArrayList<Double>();
			for (int i = 0; i < minmax.length; i++) {
				allval.add(minmax[i][j]+i);
			}
			allval.add((minmax[0][j]+minmax[1][j])/2.0);
			for (int i = 0; i < classmean.length; i++) {
				allval.add(classmean[i][j]);
			}
			allval.add((classmean[0][j]+classmean[1][j])/2.0);
			allval.add((classmean[0][j]+classmean[1][j]+overallmean[j])/3.0);
			allval.add(overallmean[j]);
			allval.add((minmax[0][j]+overallmean[j])/2.0);
			allval.add((minmax[1][j]+overallmean[j])/2.0);
			Collections.sort(allval);
			bin[j][0][0]=allval.get(0);
			for (int i = 1; i < allval.size()-1; i++) {
				bin[j][i-1][1]=allval.get(i);
				bin[j][i][0]=allval.get(i);
			}
			bin[j][allval.size()-2][1]=allval.get(allval.size()-1);
		}
		int category=9,spamcount=0;
		double[][][] conditionalprob= new double[featlen][category][classno];
		for (int j = 0; j < featlen; j++) {
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]=1;
				}
			}
			spamcount=0;
			for (int k = 0; k < trainlen; k++) {
				conditionalprob[j][selectBin(bin[j], featurestrain[k][j])][(int) labelstrain[k]]++;
				spamcount+=labelstrain[k];
			}
			int classlen[]={trainlen-spamcount+category,spamcount+category};
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]/=((double) (classlen[m]));//+classno*category));
				}
			}
		}
		double spam=((double) spamcount)/((double) trainlen);
		double ham=1.0-spam;
		if(!mode)
		{
			acc[0]=evaluateHist(featurestrain, labelstrain, conditionalprob, bin, cc[0], spam, ham, trainlen,0.0);
			acc[1]=evaluateHist(featurestest, labelstest, conditionalprob, bin, cc[1], spam, ham, testlen,0.0);
		}
		else
		{
			return ROCHist(featurestest, labelstest, conditionalprob, bin, spam, ham, testlen);
		}
		return null;
	}
	private static void chooseModel(int i,double[][]cc,double[] acc)
	{
		if(i==0)
		{
			Bernoulli(cc, acc,false);
		}
		else if(i==1)
		{
			Gaussian(cc, acc,false);
		}
		else if(i==2)
		{
			HistFour(cc, acc,false);
		}
		else
		{
			HistNine(cc, acc,false);
		}
	}
	public static void main(String[] args) {

		readData();
		Double[] mn = getMean(features, len, featlen);
		Double[] std = getStd(features, len, featlen, mn);
		//normalize(features, len, featlen, mn, std);
		globalmean=getMean(features, len, featlen);
		//randSplits(10);
		int folds=10,model=4;
		uniformSplits(folds);
		double[][][][] classconfusion=new double[folds][model][2][4];
		double[][][] accuracy=new double[folds][model][2];
		double[][] foldavgaccuracy=new double[model][2];
		double[][][] foldtprfpr=new double[model][2][2];
		double[] avgaccuracy=new double[2];
		for (int i = 1; i <=folds ; i++) {
			pickKfold(i);
			System.out.println("Fold: "+i);
			for (int j = 0; j < model; j++) {
				System.out.println("Model NO: "+(j+1));
				chooseModel(j,classconfusion[i-1][j] , accuracy[i-1][j]);
				for (int k = 0; k < 2; k++) {
					foldavgaccuracy[j][k]+=accuracy[i-1][j][k];
					System.out.println("Accuracy: "+accuracy[i-1][j][k]+" Error Rate: "+(1-accuracy[i-1][j][k]));
					double fpr=classconfusion[i-1][j][k][1]/(classconfusion[i-1][j][k][1]+classconfusion[i-1][j][k][2]);
					double fnr=classconfusion[i-1][j][k][3]/(classconfusion[i-1][j][k][3]+classconfusion[i-1][j][k][0]);
					foldtprfpr[j][k][0]+=fpr;
					foldtprfpr[j][k][1]+=fnr;
					System.out.println("False Positive Rate: "+fpr+" False Negative Rate: "+fnr);
				}
				System.out.println();
			}
		}
		System.out.println("\nOverall\n");
		for (int i = 0; i < model; i++) {
			System.out.println("Model NO: "+(i+1));
			for (int j = 0; j < 2; j++) {
				foldavgaccuracy[i][j]/=folds;
				foldtprfpr[i][j][0]/=folds;
				foldtprfpr[i][j][1]/=folds;
				System.out.println("Overall Accuracy: "+foldavgaccuracy[i][j]+"Overall Error Rate: "+(1-foldavgaccuracy[i][j]));
				System.out.println("Overall False Positive Rate: "+foldtprfpr[i][j][0]+" Overall False Negative Rate: "+foldtprfpr[i][j][1]);
			}
		}
		pickKfold(10);
		double broc[][]=Bernoulli(new double[1][1],new double[1],true);
		try {
			FileWriter cclgr = new FileWriter("resulthw3/nb_bernoulli.txt");
			BufferedWriter cclgrbw = new BufferedWriter(cclgr);
			for (int i = 0; i < broc.length; i++) {
				cclgrbw.write(i+" "+broc[i][0]+" "+broc[i][1]+" "+broc[i][2]+" "+broc[i][3]+"\n");
			}
			cclgrbw.close();
			cclgr.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		broc=Gaussian(new double[1][1],new double[1],true);
		try {
			FileWriter cclgr = new FileWriter("resulthw3/nb_gaussian.txt");
			BufferedWriter cclgrbw = new BufferedWriter(cclgr);
			for (int i = 0; i < broc.length; i++) {
				cclgrbw.write(i+" "+broc[i][0]+" "+broc[i][1]+" "+broc[i][2]+" "+broc[i][3]+"\n");
			}
			cclgrbw.close();
			cclgr.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		/*System.out.println(broc.length);
		  for (int i = 0; i < broc.length; i++) {
			System.out.println(broc[i][0]+" "+broc[i][1]+" "+broc[i][2]+" "+broc[i][3]+"      "+(broc[i][0]/(broc[i][0]+broc[i][3]))+" & "+(broc[i][1]/(broc[i][1]+broc[i][2])));
		}*/
		broc=HistFour(new double[1][1],new double[1],true);
		try {
			FileWriter cclgr = new FileWriter("resulthw3/nb_hist4.txt");
			BufferedWriter cclgrbw = new BufferedWriter(cclgr);
			for (int i = 0; i < broc.length; i++) {
				cclgrbw.write(i+" "+broc[i][0]+" "+broc[i][1]+" "+broc[i][2]+" "+broc[i][3]+"\n");
			}
			cclgrbw.close();
			cclgr.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		broc=HistNine(new double[1][1],new double[1],true);
		try {
			FileWriter cclgr = new FileWriter("resulthw3/nb_hist9.txt");
			BufferedWriter cclgrbw = new BufferedWriter(cclgr);
			for (int i = 0; i < broc.length; i++) {
				cclgrbw.write(i+" "+broc[i][0]+" "+broc[i][1]+" "+broc[i][2]+" "+broc[i][3]+"\n");
			}
			cclgrbw.close();
			cclgr.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}

}
