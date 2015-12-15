package hw5.features.regression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Jama.Matrix;
import edu.uci.lasso.LassoFit;
import edu.uci.lasso.LassoFitGenerator;

public class RegularLR {

	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int pollutetrainlen=4140;
	static int pollutetestlen=461;
	static int pollutefeatlen=1057;
	static int trainlen,testlen;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static ArrayList<double[][]> featklist = new ArrayList<double[][]>();
	static ArrayList<double[]> labelsklist = new ArrayList<double[]>();
	static ArrayList<Integer> indexlist = new ArrayList<Integer>();
	static double[][] featurestrain = new double[3680][featlen];
	static double [] labelstrain = new double[3680];
	static double [][] featurestest= new double[921][featlen];
	static double [] labelstest= new double[921];
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
	private static void readPolluted(){
		int one=0,zero=0;
		trainlen=pollutetrainlen;
		testlen=pollutetestlen;
		featlen=pollutefeatlen;
		featurestrain = new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest = new double[testlen][featlen];
		labelstest  = new double[testlen];
		try {
			FileReader featureread = new FileReader("data/spam_polluted/train_feature.txt");
			FileReader labelsread = new FileReader("data/spam_polluted/train_label.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			BufferedReader labelsreadbr = new BufferedReader(labelsread);
			String sCurrentLine,sLabel;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sLabel = labelsreadbr.readLine();
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					featurestrain[ind][i]=Double.parseDouble(feats[i]);

				}
				labelstrain[ind]=Double.parseDouble(sLabel);
				if(labelstrain[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			labelsreadbr.close();
			featurereadbr.close();
			labelsread.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
		try {
			FileReader featureread = new FileReader("data/spam_polluted/test_feature.txt");
			FileReader labelsread = new FileReader("data/spam_polluted/test_label.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			BufferedReader labelsreadbr = new BufferedReader(labelsread);
			String sCurrentLine,sLabel;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sLabel = labelsreadbr.readLine();
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					featurestest[ind][i]=Double.parseDouble(feats[i]);

				}
				labelstest[ind]=Double.parseDouble(sLabel);
				if(labelstest[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			labelsreadbr.close();
			featurereadbr.close();
			labelsread.close();
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
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
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
	private static double hofw(double w[],double x[],double y){
		double ans=0.0;
		for (int i = 0; i < w.length; i++) {
			ans+=w[i]*x[i];
			//System.out.println(ans+"    "+w[i]+"    "+x[i]);
		}
		//ans+=w0;
		//System.out.println(ans);
		ans=y-(1.0/(1.0+Math.exp(-ans)));
		return ans;
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
	private static void normalize(double[][] features,int x,int y,Double[] mean,Double[] std){
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				features[j][i]-=mean[i];
				features[j][i]/=std[i];
			}
		}
	}
	private static void noramalizeboth(){
		Double[] mean=getMean(featurestrain, trainlen, featlen);
		Double[] std=getStd(featurestrain, trainlen, featlen, mean);
		normalize(featurestrain, trainlen, featlen, mean, std);
		normalize(featurestest, testlen, featlen, mean, std);
	}
	private static double[][] addColumn(double[][] features,double val) {
		double temp[][]=new double[features.length][featlen];
		for(int i=0;i<features.length;i++)
		{
			for (int j = 0; j < featlen-1; j++) {
				temp[i][j]=features[i][j];	
			}
			temp[i][featlen-1]=val;
		}
		return temp;
	}
	private static double[][] removeColumn(double[][] features) {
		double temp[][]=new double[features.length][featlen];
		for(int i=0;i<features.length;i++)
		{
			for (int j = 0; j < featlen; j++) {
				temp[i][j]=features[i][j];	
			}
		}
		return temp;
	}
	private static void LogisticReg(boolean ridge){
		noramalizeboth();
		featlen++;
		featurestrain=addColumn(featurestrain,1.0);
		featurestest=addColumn(featurestest,1.0);
		double lambda=0.003;
		double w[]=new double[featlen];
		for (int ij = 0; ij < featlen; ij++) {
			w[ij]=0.1;
		}
		double gte=100000.0,cte=0.0,ctr=0.0,thresh=0.000001,tacc=0.0,tracc=0.0,septhresh=0.5,temptracc=0.0,temptacc=0.0;
		double control=0.0;
		if(ridge)
			control=0.00015;
		Matrix x = new Matrix(featurestrain);
		Matrix y = new Matrix(labelstrain,1);
		y=y.transpose();
		Matrix xtest=new Matrix(featurestest);
		Matrix ytest = new Matrix(labelstest,1);
		ytest=ytest.transpose();
		while(true)
		{
			for (int ij = 0; ij < trainlen; ij++) {
				double temp=hofw(w,featurestrain[ij],labelstrain[ij]);
				for (int j = 0; j < featlen; j++) {
					w[j]=w[j]+lambda*temp*featurestrain[ij][j]+control*w[j];
				}
				//w[featlen]=w[featlen]+lambda*hofw(w,featurestrain[ij],labelstrain[ij]);
			}
			Matrix wma=new Matrix(w, w.length);
			Matrix mul3 = xtest.times(wma);

			for(int ij=0;ij<mul3.getRowDimension();ij++)
			{
				mul3.set(ij, 0, 1/(1+Math.exp(-mul3.get(ij, 0))));
			}

			Matrix error= ytest.minus(mul3);
			tacc=0.0;
			for (int j = 0; j < mul3.getRowDimension(); j++) {
				if( Math.floor(mul3.get(j, 0)+septhresh) == ytest.get(j, 0) )
				{
					tacc++;
				}
			}
			tacc/=(double) mul3.getRowDimension();
			tacc*=100;
			Matrix fete=(error.transpose().times(error)).times(1);
			System.out.println("Testing error:"+fete.get(0, 0)/((double) (testlen)));
			System.out.println("testing Accuracy:"+tacc);
			cte=fete.get(0, 0)/((double) (testlen));
			Matrix mul4 = x.times(wma);

			for(int ij=0;ij<mul4.getRowDimension();ij++)
			{
				mul4.set(ij, 0, 1/(1+Math.exp(-mul4.get(ij, 0))));
			}
			Matrix errortrain= y.minus(mul4);
			tracc=0.0;
			for (int j = 0; j < mul4.getRowDimension(); j++) {
				if( Math.floor(mul4.get(j, 0)+septhresh) == y.get(j, 0) )
				{
					tracc++;
				}
			}
			tracc/=(double) mul4.getRowDimension();
			tracc*=100;
			Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
			System.out.println("Training error:"+fet.get(0, 0)/((double) (trainlen)));
			System.out.println("training Accuracy:"+tracc);
			ctr=fet.get(0, 0)/((double) (trainlen));
			if(Math.abs(cte-gte)<thresh)
			{
				//System.out.println(Math.abs(cte-gte)+"  ");
				break;
			}
			//if(cte>gte)
			//break;
			else
			{
				gte=cte;
			}

		}
		System.out.println("over  \n");
		int cctrind=0,cctind=0;
		ArrayList<Double[]> confusiontc= new ArrayList<Double[]>();
		ArrayList<Double[]> confusiontrc= new ArrayList<Double[]>();
		Matrix wma=new Matrix(w, w.length);
		Matrix mul3 = (xtest.times(wma));
		for(int ij=0;ij<mul3.getRowDimension();ij++)
		{
			mul3.set(ij, 0, 1/(1+Math.exp(-mul3.get(ij, 0))));
		}
		Matrix error= ytest.minus(mul3);
		Matrix fete=(error.transpose().times(error)).times(1);
		septhresh=0.0;
		for(int k=0; k < mul3.getRowDimension() ; k++)
		{
			septhresh=Math.min(septhresh, mul3.get(k, 0));
		}
		septhresh-=1;
		for(int k=0; k < mul3.getRowDimension()+1 ; k++)
		{
			if(k>0)
				septhresh=mul3.get(k-1, 0);
			temptacc=0.0;
			Double[] tcc={0.0,0.0,0.0,0.0};
			for (int j = 0; j < mul3.getRowDimension(); j++) {
				if(mul3.get(j, 0)>septhresh)
				{
					if(ytest.get(j, 0)==1)
					{
						tcc[0]++;
						temptacc++;
					}
					else
					{
						tcc[1]++;	
					}
				}
				else
				{
					if(ytest.get(j, 0)==0)
					{
						tcc[2]++;
						temptacc++;
					}
					else
					{
						tcc[3]++;	
					}
				}
			}
			temptacc/=(double) mul3.getRowDimension();
			temptacc*=100;
			confusiontc.add(k,tcc);
			if(tacc<temptacc)
			{
				tacc=temptacc;
				cctind=k;
			}
		}
		System.out.println("Testing error:"+fete.get(0, 0)/(double) testlen);
		System.out.println("testing Accuracy:"+tacc);
		Matrix mul4 = (x.times(wma));
		for(int ij=0;ij<mul4.getRowDimension();ij++)
		{
			mul4.set(ij, 0, 1/(1+Math.exp(-mul4.get(ij, 0))));
		}
		Matrix errortrain= y.minus(mul4);
		Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
		septhresh=0.0;
		for(int k=0; k < mul4.getRowDimension() ; k++)
		{
			septhresh=Math.min(septhresh, mul4.get(k, 0));
		}
		septhresh-=1;
		for(int k=0; k < mul4.getRowDimension()+1 ; k++)
		{
			if(k>0)
				septhresh=mul4.get(k-1, 0);
			temptracc=0.0;
			Double[] trcc={0.0,0.0,0.0,0.0};
			for (int j = 0; j < mul4.getRowDimension(); j++) {
				if( Math.floor(mul4.get(j, 0)+septhresh) == 1)
				{
					if(y.get(j, 0)==1)
					{
						trcc[0]++;
						temptracc++;
					}
					else
					{
						trcc[1]++;	
					}
				}
				else
				{
					if(y.get(j, 0)==0)
					{
						trcc[2]++;
						temptracc++;
					}
					else
					{
						trcc[3]++;	
					}
				}
			}
			temptracc/=(double) mul4.getRowDimension();
			temptracc*=100;
			confusiontrc.add(k,trcc);
			if(tracc<temptracc)
			{
				tracc=temptracc;
				cctrind=k;
			}
		}
		System.out.println("Training error:"+fet.get(0, 0)/(double) trainlen);
		System.out.println("training Accuracy:"+tracc);
		featlen--;
		featurestrain=removeColumn(featurestrain);
		featurestest=removeColumn(featurestest);

	}
	private static float[] getFloat(double[] feat){
		float[] res=new float[feat.length];
		for (int i = 0; i < res.length; i++) {
			res[i]=(float) feat[i];
		}
		return res;
	}
	private static void Lasso(){
		try {
			FileWriter fw = new FileWriter("resulthw5/train.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < trainlen; i++) {
				bw.write(((int)labelstrain[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestrain[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		try {
			FileWriter fw = new FileWriter("resulthw5/test.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < testlen; i++) {
				bw.write(((int)labelstest[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestest[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}
	public static void main(String[] args) {

		readPolluted();
		Lasso();
		
		//LogisticReg(false);
        //noramalizeboth();
		/*LassoFitGenerator fitGenerator = new LassoFitGenerator();
		int numObservations = featurestrain.length;
		try {
			fitGenerator.init(featlen, numObservations);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (int i = 0; i < numObservations; i++) {
			fitGenerator.setObservationValues(i, getFloat(featurestrain[i]));
			fitGenerator.setTarget(i, labelstrain[i]);
		}


		LassoFit fit = fitGenerator.fit(-1);

		System.out.println(fit);
		for (int i = 0; i < fit.intercepts.length; i++) {
			System.out.println(fit.intercepts[i]);
		}

		 System.out.println(fit.numberOfLambdas+" "+fit.intercepts.length+" "+fit.nonZeroWeights[fit.numberOfLambdas-1]+" "+fit.compressedWeights.length+"  "+fit.compressedWeights[0].length);
		 featlen++;
		 featurestrain=addColumn(featurestrain, 1.0);
		 featurestest=addColumn(featurestest, 1.0);
		 Matrix x = new Matrix(featurestrain);
		 Matrix y = new Matrix(labelstrain,1);
		 y=y.transpose();
		 Matrix xtest=new Matrix(featurestest);
		 Matrix ytest = new Matrix(labelstest,1);
		 ytest=ytest.transpose();
		 for (int k = 0; k < fit.numberOfLambdas; k++) {
			 System.out.println("Iter: "+k);
			 double[] w = new double[featlen];
			 double[] dw = fit.getWeights(k);
			 for (int i = 0; i < w.length-1; i++) {
				 w[i]=dw[i];
			 }
			 w[w.length-1]=fit.intercepts[k];

			 Matrix wma=new Matrix(w, w.length);
			 System.out.println(wma.getColumnDimension()+"  "+wma.getRowDimension());
			 System.out.println(xtest.getColumnDimension()+"  "+xtest.getRowDimension());
			 Matrix mul3 = xtest.times(wma);

			 for(int ij=0;ij<mul3.getRowDimension();ij++)
			 {
				 mul3.set(ij, 0, 1/(1+Math.exp(-mul3.get(ij, 0))));
			 }

			 Matrix error= ytest.minus(mul3);
			 double tacc = 0.0;
			 for (int j = 0; j < mul3.getRowDimension(); j++) {
				 if( Math.floor(mul3.get(j, 0)+0.5) == ytest.get(j, 0) )
				 {
					 tacc++;
				 }
			 }
			 tacc/=(double) mul3.getRowDimension();
			 tacc*=100;
			 Matrix fete=(error.transpose().times(error)).times(1);
			 System.out.println("Testing error:"+fete.get(0, 0)/((double) (testlen)));
			 System.out.println("testing Accuracy:"+tacc);
			 Matrix mul4 = x.times(wma);

			 for(int ij=0;ij<mul4.getRowDimension();ij++)
			 {
				 mul4.set(ij, 0, 1/(1+Math.exp(-mul4.get(ij, 0))));
			 }
			 Matrix errortrain= y.minus(mul4);
			 double tracc = 0.0;
			 for (int j = 0; j < mul4.getRowDimension(); j++) {
				 if( Math.floor(mul4.get(j, 0)+0.5) == y.get(j, 0) )
				 {
					 tracc++;
				 }
			 }
			 tracc/=(double) mul4.getRowDimension();
			 tracc*=100;
			 Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
			 System.out.println("Training error:"+fet.get(0, 0)/((double) (trainlen)));
			 System.out.println("training Accuracy:"+tracc);*/
		 //}
	}


}
