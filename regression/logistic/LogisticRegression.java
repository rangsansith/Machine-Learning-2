package hw2.regression.logistic;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Jama.Matrix;

public class LogisticRegression {

	private static double hofw(double w[],double x[],double y){
		double ans=0.0;
		for (int i = 0; i < w.length; i++) {
			ans+=w[i]*x[i];
			//System.out.println(ans+"    "+w[i]+"    "+x[i]);
		}
		//ans+=w0;
		//System.out.println(ans);
		ans=y-(1/(1+Math.exp(-ans)));
		return ans;
	}
	public static void main(String[] args) {
		double[][] features = new double[4601][58];
		double [][] labels = new double[4601][1];
		ArrayList<Integer> indexlist = new ArrayList<Integer>();
		HashMap<Integer, Boolean> seen = new HashMap<Integer, Boolean>();
		FileReader featureread;
		try {
			featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				indexlist.add(ind,ind);
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);
				}
				features[ind][feats.length-1]=1.0;
				labels[ind][0]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		ArrayList<double[][]> featklist = new ArrayList<double[][]>();
		ArrayList<double[][]> labelsklist = new ArrayList<double[][]>();
		int kfold=1001;int listno=0;
		Random n = new Random(); 
		while(indexlist.size()!=0){
			int r=Math.min(kfold, indexlist.size());
			double[][] featurestemp = new double[r][58];
			double [][] labelstemp = new double[r][1];
			for (int i = 0; i < kfold; i++) {
				if(indexlist.size()==0)
					break;
				int ind=n.nextInt(indexlist.size());
				int tmpind=indexlist.get(ind);
				if(seen.containsKey(tmpind))
					System.out.println("ayayo");
				else
					seen.put(tmpind, true);
				for (int j = 0; j < 57; j++) {
					featurestemp[i][j]=features[tmpind][j];
				}
				labelstemp[i][0]=labels[tmpind][0];
				indexlist.remove(ind);
			}
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
		int nind=0;Double msetr=0.0,msete=0.0;
		for (int i = 0; i < listno; i++) {
			nind=0;
			int fte=featklist.get(i).length;
			int ftr=4601-fte;
			double[][] featurestrain = new double[ftr][58];
			double [][] labelstrain = new double[ftr][1];
			double[][] featurestest = new double[fte][58];
			double [][] labelstest = new double[fte][1];
			double [] mean = new double[58];
			double [] std = new double[58];
			for (int j = 0; j < listno; j++) {
				double[][] featurestemp = featklist.get(j);
				double [][] labelstemp = labelsklist.get(j);
				//System.out.println("before "+i+"  "+j+"  "+nind+"  "+labelsklist.get(j).size());
				if(j==i)
				{
					featurestest = featklist.get(j);
					labelstest = labelsklist.get(j);
				}
				else
				{
					for (int k = 0; k < labelstemp.length; k++) {
						labelstrain[nind][0]=labelstemp[k][0];
						for (int k2 = 0; k2 < 57; k2++) {
							featurestrain[nind][k2]=featurestemp[k][k2];
							mean[k2]+=featurestrain[nind][k2];
						}
						nind++;
					}
				}
			}
			for(int ij=0;ij<57;ij++)
			{
				mean[ij]/=nind;
			}
			for (int j = 0; j < mean.length-1; j++) {
				std[j]=0.0;
				for (int j2 = 0; j2 < nind; j2++) {
					std[j]+=(featurestrain[j2][j]-mean[j])*(featurestrain[j2][j]-mean[j]);    	
				}
				std[j]=Math.sqrt(std[j]);
			}
			for (int j = 0; j < nind; j++) {
				for (int j2 = 0; j2 < mean.length-1; j2++) {
					featurestrain[j][j2]-=mean[j2]; 
					featurestrain[j][j2]/=std[j2]; 
				}
			}
			for (int j = 0; j < 4601-nind; j++) {
				for (int j2 = 0; j2 < mean.length-1; j2++) {
					featurestest[j][j2]-=mean[j2]; 
					featurestest[j][j2]/=std[j2]; 
				}
			}
			nind=labelsklist.get(i).length;
			double w[]=new double[58];
			double lambda=0.0000001;
			for (int ij = 0; ij < w.length; ij++) {
				w[ij]=0.001;
			}
			//w[58]=2.2;
			double gte=100000.0,gtr=100000.0,cte=0.0,ctr=0.0,thresh=0.0000001,tacc=0.0,tracc=0.0,septhresh=0.5,w0=1000.8;
			Matrix x = new Matrix(featurestrain);
			Matrix y = new Matrix(labelstrain);
			Matrix xtest=new Matrix(featurestest);
			Matrix ytest = new Matrix(labelstest);
			while(true)
			{
				for (int ij = 0; ij < featurestrain.length; ij++) {
					
					for (int j = 0; j < w.length; j++) {
						//System.out.println("before "+i+"  "+j+"  "+w[j]);
						double t=w[j];
						w[j]=w[j]-lambda*hofw(w,featurestrain[ij],labelstrain[ij][0])*featurestrain[ij][j];
						//System.out.println(j+"  "+lambda*hofw(w,featurestrain[ij],labelstrain[ij][0])*featurestrain[ij][j]);
						//System.out.println("after "+i+"  "+j+"  "+w[j]);
						//if(t!=w[j])
						//System.out.println(t+" to "+w[j]);
					}
				}
				//for (int j = 0; j < w.length; j++) {
				//		System.out.println(w[i]);
				//}
				//Matrix lam=Matrix.identity(58, 58).times(lambda);
				Matrix wma=new Matrix(w, w.length);
				Matrix mul3 = xtest.times(wma);

				for(int ij=0;ij<mul3.getRowDimension();ij++)
				{
					mul3.set(ij, 0, 1/(1+Math.exp(-mul3.get(ij, 0))));
				}

				Matrix error= ytest.minus(mul3);
				tacc=0.0;
				for (int j = 0; j < mul3.getRowDimension(); j++) {
					//System.out.println(Math.floor(mul3.get(j, 0)+septhresh)+" test  "+ytest.get(j, 0));
					if( Math.floor(mul3.get(j, 0)+septhresh) == ytest.get(j, 0) )
					{
						tacc++;
					}
				}
				tacc/=(double) mul3.getRowDimension();
				tacc*=100;
				Matrix fete=(error.transpose().times(error)).times(1);
				//fete.print(1, 6);
				System.out.println("Iteration no:"+(i+1)+"  "+labelstrain.length+"  "+labelstest.length);
				System.out.println("Testing error:"+fete.get(0, 0)/((double) (nind)));
				System.out.println("testing Accuracy:"+tacc);
				cte=fete.get(0, 0)/((double) (nind));
				//error.print(1, 6);
				Matrix mul4 = x.times(wma);
				
				for(int ij=0;ij<mul4.getRowDimension();ij++)
				{
					mul4.set(ij, 0, 1/(1+Math.exp(-mul4.get(ij, 0))));
				}
				Matrix errortrain= y.minus(mul4);
				tracc=0.0;
				for (int j = 0; j < mul4.getRowDimension(); j++) {
					//System.out.println(Math.floor(mul3.get(0, j)+septhresh)+" train  "+ytest.get(0, j));
					if( Math.floor(mul4.get(j, 0)+septhresh) == y.get(j, 0) )
					{
						tracc++;
					}
				}
				tracc/=(double) mul4.getRowDimension();
				tracc*=100;
				Matrix fet=(errortrain.transpose().times(errortrain)).times(1);
				//fet.print(1, 6);
				System.out.println("Training error:"+fet.get(0, 0)/((double) (4601-nind)));
				System.out.println("training Accuracy:"+tracc);
				ctr=fet.get(0, 0)/((double) (4601-nind));
				//ctr=fet.get(0, 0)/nind;
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
					gtr=ctr;
				}
				
			}
			System.out.println("Iteration no:"+(i+1));
			System.out.println("Testing error:"+gte);
			System.out.println("Training error:"+gtr);
			msetr+=gtr;
			msete+=gte;

		}
		System.out.println("Average testing error:"+(msete/((double) listno-0)));
		System.out.println("Average training error:"+(msetr/((double) listno-0)));

	}

}
