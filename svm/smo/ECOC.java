package hw6.svm.smo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

public class ECOC {

	static int featlen=200;
	static int classno=10;
	static int trainlen=2000,testlen=10000;
	static int codesize=50;
	static int stop=2000;
	static double [] labelstrain = new double[trainlen];
	static double [] labelstest= new double[testlen];
	static double [] codelabelstrain = new double[trainlen];
	static double [] codelabelstest= new double[testlen];
	static double [] predictiontest;
	static double [] predictiontrain;
	static double[][] totalpredictiontest = new double[codesize][testlen];
	static double[][] totalpredictiontrain = new double[codesize][trainlen];
	static double[][] featurestrain = new double[trainlen][featlen];
	static double[][] featurestest = new double[testlen][featlen];
	static double [][] kernelmat= new double[1][1];
	static ArrayList<double[]> codelabels = new ArrayList<double[]>();
	static double[] alpha;
	static double b;
	static double regc=10;
	static double numtol=0.001;
	static int maxpass=10;
	static int kerneltype=2;
	static double sigmag=1.0/featlen;	
	static double tanc=0.1;
	static double tanb=0.0;
	private static double tol2 = 0.001;
	private static double [] E;
	private static void readData()
	{
		try {
			FileReader featureread = new FileReader("data/digitsdataset/trainecoc.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					featurestrain[ind][i]=Double.parseDouble(feats[i]);	
				}
				labelstrain[ind]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/digitsdataset/testecoc.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					featurestest[ind][i]=Double.parseDouble(feats[i]);	
				}
				labelstest[ind]=Double.parseDouble(feats[feats.length-1]);
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
				FileReader featureread = new FileReader("data/digitsdataset/codes");
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
		HashMap<Double, Boolean> code= new HashMap<Double,Boolean>();
		double val=0.0;
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
				System.out.println(i+"  "+s+"  "+b+" "+val);
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
				FileWriter fwc = new FileWriter("data/digitsdataset/codes");
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
	private static void init(int codenum){
		alpha=new double[trainlen];
		initalphas();
		b=0;
		for (int i = 0; i < testlen; i++) {
			codelabelstest[i]=codelabels.get((int) labelstest[i])[codenum];
		}
		codelabelstest=changeleabels(codelabelstest);
		for (int i = 0; i < trainlen; i++) {
			codelabelstrain[i]=codelabels.get((int) labelstrain[i])[codenum];
		}
		codelabelstrain=changeleabels(codelabelstrain);
		predictiontest= new double[testlen];
		predictiontrain= new double[trainlen];
	}
	private static double[] changeleabels(double[] labelsc){
		for (int i = 0; i < labelsc.length; i++) {
			if(labelsc[i]==0)
				labelsc[i]=-1;
		}
		return labelsc;
	}
	private static void initalphas(){
		for (int i = 0; i < trainlen; i++) {
			alpha[i]=0.0;
		}
	}
	private static double dotproduct(double[] i,double[] j){
		double ans=0;
		for(int k=0;k<featlen;k++)
		{
			ans+=i[k]*j[k];
		}
		return ans;
	}
	private static double euclidist(double[] i, double[] j) {
		double sum=0;
		for (int k=0;k<featlen;k++) {
			sum+=(i[k]-j[k])*(i[k]-j[k]);
		}
		return sum;
	}
	private static double kernel(double[] i, double[] j){
		if(kerneltype==1)
		{
			return dotproduct(i, j);
		}
		else if(kerneltype==2)
		{
			return Math.exp(-(euclidist(i, j)*(sigmag*sigmag)));
		}
		else if(kerneltype==3)
		{
			return Math.tanh(sigmag*dotproduct(i, j)+tanb);
		}
		else
		{
			return Math.pow(sigmag*dotproduct(i, j)+tanb, tanc);
		}
	}
	private static void setkernel(){
		for(int i=0;i<trainlen;i++)
		{
			for(int j=0;j<trainlen;j++)
			{
				kernelmat[i][j]=kernel(featurestrain[i],featurestrain[j]);
			}
		}
	}
	private static double mpredict(int xno){
		double ans=b;
		double []kern=kernelmat[xno];
		for (int i = 0; i < trainlen; i++) {
			ans+=alpha[i]*kern[i]*codelabelstrain[i];
		}
		return ans;
	}
	private static double mpredict(double [] xno){
		double ans=b;
		for (int i = 0; i < trainlen; i++) {
			ans+=alpha[i]*kernel(xno,featurestrain[i])*codelabelstrain[i];
		}
		return ans;
	}
	private static double smoObj(double alphai,double alphaj,double aj, int yi, int yj, double kij, double kii, double kjj, double vi, double vj) {
		double s = yi*yj;
		double gamma = alphai + s*alphaj;
		return (gamma + (1-s)*aj - 0.5*kii*(gamma-s*aj)*(gamma-s*aj) - 0.5*kjj*aj*aj +
				- s*kij*(gamma-s*aj)*aj - yi*(gamma-s*aj)*vi - yj*aj*vj);
	}
	private static double[] LH(int i,int j){
		double[] ans={0,0};
		if(codelabelstrain[i]==codelabelstrain[j])
		{
			ans[0]=Math.max(0, alpha[j]+alpha[i]-regc);
			ans[1]=Math.min(regc,alpha[j]+alpha[i]);
		}
		else
		{
			ans[0]=Math.max(0, alpha[j]-alpha[i]);
			ans[1]=Math.min(regc, regc+alpha[j]-alpha[i]);
		}
		return ans;
	}
	private static void setcache(){
		double error=0;
		for (int i = 0; i < trainlen; i++) {
			error=mpredict(i)-labelstrain[i];
			E[i]=error;
		}
	}
	private static void writeforlib(){

		try {
			FileWriter fw = new FileWriter("resulthw6/trainlib.txt");
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
			FileWriter fw = new FileWriter("resulthw6/testlib.txt");
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
	private static int update(int i,int j,double ei){
		double ej=mpredict(j)-codelabelstrain[j];
		double alphai = alpha[i];
		double alphaj = alpha[j];
		double[] LH = LH(i, j);
		if(LH[0]==LH[1])
			return 0;
		double eta=2*kernelmat[i][j]-kernelmat[i][i]-kernelmat[j][j];
		if(eta>=0)
			return 0;
		alpha[j]=alpha[j]-codelabelstrain[j]*(ei-ej)/eta;
		if(alpha[j]>LH[1])
			alpha[j]=LH[1];
		else if(alpha[j]<LH[0])
			alpha[j]=LH[0];
		if(Math.abs(alphaj-alpha[j])<0.00001)
			return 0;
		alpha[i]=alpha[i]+codelabelstrain[i]*codelabelstrain[j]*(alphaj-alpha[j]);
		double b1=b-ei-codelabelstrain[i]*(alpha[i]-alphai)*kernelmat[i][i]-codelabelstrain[j]*(alpha[j]-alphaj)*kernelmat[i][j];
		double b2=b-ej-codelabelstrain[i]*(alpha[i]-alphai)*kernelmat[i][j]-codelabelstrain[j]*(alpha[j]-alphaj)*kernelmat[j][j];
		if(alpha[i]<regc)
			b=b1;
		else if(alpha[j]<regc)
			b=b2;
		else
			b=(b1+b2)/2;
		return 1;
	}
	private static int ExamineExample(int j) {
		int i = 0;
		int randpos;
		int yj = (int) codelabelstrain[j];
		double alphaj = alpha[j];
		double ej = E[j];
		double rj = ej*yj;
		if ((rj<-numtol && alphaj<regc) || (rj>numtol && alphaj>0)) {
			boolean exists = false;
			for (int k=0; k<trainlen; k++)
				if (alpha[k]>0 && alpha[k]<regc) {
					exists = true;
					break;
				}
			if (exists) {
				//second choice heuristics:
				int maxind = 0;
				double maxval = Math.abs(E[0]-ej);
				for (int k=1; k<trainlen; k++)
					if (Math.abs(E[k]-ej) > maxval) {
						maxval = Math.abs(E[k]-ej);
						maxind = k;
					}
				if (update(maxind, j) == 1)
					return 1;
			}
			//loop over non-zero & non-C alpha, starting at a random point:
			randpos = (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = (randpos+k)%trainlen;
				if (alpha[i]>0 && alpha[i]<regc) {
					if (update(i, j) == 1)
						return 1;
				}
			}
			//loop over all i, starting at a random point
			randpos = (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = (randpos+k)%trainlen;
				if (update(i, j) == 1)
					return 1;
			}
		}
		return 0;
	}
	private static int anotherheur(int j) {
		int i = 0;
		int randpos;
		int yj = (int) codelabelstrain[j];
		double alphaj = alpha[j];
		double ej = E[j];
		double rj = ej*yj;
		if ((rj<-numtol && alphaj<regc) || (rj>numtol && alphaj>0)) {
			boolean exists = false;
			for (int k=0; k<trainlen; k++)
				if (alpha[k]>0 && alpha[k]<regc) {
					exists = true;
					break;
				}
			if (exists) {
				//second choice heuristics:
				int maxind = 0;
				double maxval = Math.abs(E[0]-ej);
				for (int k=1; k<trainlen; k++)
					if (Math.abs(E[k]-ej) > maxval) {
						maxval = Math.abs(E[k]-ej);
						maxind = k;
					}
				if (updatenew(maxind, j) == 1)
					return 1;
			}
			//loop over non-zero & non-C alpha, starting at a random point:
			randpos =0;// (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = k;
				if (alpha[i]>0 && alpha[i]<regc) {
					if (updatenew(i, j) == 1)
						return 1;
				}
			}
			//loop over all i, starting at a random point
			randpos = 0;//(int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = k;
				if (updatenew(i, j) == 1)
					return 1;
			}
		}
		return 0;
	}
	private static int update(int i, int j) {
		double ai, aj;
		if (i == j) return 0;
		double bb = b;
		double alphai = alpha[i];
		double alphaj = alpha[j];
		int yi = (int) codelabelstrain[i];
		int yj = (int) codelabelstrain[j];
		double ei = E[i];
		double ej = E[j];
		double s = yi*yj;
		double[] LH = LH(i, j);
		double L = LH[0];
		double H = LH[1];
		if (L == H)
			return 0;
		double kii = kernelmat[i][i];
		double kjj = kernelmat[j][j];
		double kij = kernelmat[i][j]; 
		double eta = 2*kij-kii-kjj;
		if (eta < 0) {
			aj = alphaj - yj*(ei-ej)/eta;
			if (aj < L)
				aj = L;
			else if (aj > H)
				aj = H;
		} else {
			double vi, vj;
			vi = mpredict(i) - yi*alphai*kii - yj*alphaj*kij;
			vj = mpredict(j) - yi*alphai*kij - yj*alphaj*kjj;
			double Lobj = smoObj(alphai,alphaj,L, yi, yj, kij, kii, kjj, vi, vj);
			double Hobj = smoObj(alphai,alphaj,H, yi, yj, kij, kii, kjj, vi, vj);
			if (Lobj > Hobj+tol2)
				aj = L;
			else if (Lobj < Hobj-tol2)
				aj = H;
			else
				aj = alphaj;
		}
		if (Math.abs(aj-alphaj) < tol2*(aj+alphaj+tol2))
			return 0;
		ai = alphai + s*(alphaj-aj);
		double b1=b-ei-codelabelstrain[i]*(ai-alphai)*kernelmat[i][i]-codelabelstrain[j]*(aj-alphaj)*kernelmat[i][j];
		double b2=b-ej-codelabelstrain[i]*(ai-alphai)*kernelmat[i][j]-codelabelstrain[j]*(aj-alphaj)*kernelmat[j][j];
		if(alpha[i]<regc)
			b=b1;
		else if(alpha[j]<regc)
			b=b2;
		else
			b=(b1+b2)/2;
		alpha[i] = ai;
		alpha[j] = aj;
		//updatenonbound(j,alpha[j]);
		//updatenonbound(i,alpha[i]);
		for (int k=0; k<trainlen; k++) {
			double kik = kernelmat[i][k];  
			double kjk = kernelmat[j][k];  
			E[k] += (ai-alphai)*yi*kik + (aj-alphaj)*yj*kjk - bb + b;
		}
		return 1;
	}
	private static int updatenew(int i, int j) {
		double ai, aj;
		if (i == j) return 0;
		double bb = b;
		double alphai = alpha[i];
		double alphaj = alpha[j];
		int yi = (int) codelabelstrain[i];
		int yj = (int) codelabelstrain[j];
		double ei = E[i];
		double ej = E[j];
		double s = yi*yj;
		double[] LH = LH(i, j);
		double L = LH[0];
		double H = LH[1];
		if (L == H)
			return 0;
		double kii = kernelmat[i][i];
		double kjj = kernelmat[j][j];
		double kij = kernelmat[i][j]; 
		double eta = 2*kij-kii-kjj;
		if(eta>=0)
			return 0;
		aj=alphaj-yj*(ei-ej)/eta;
		if(aj>H)
			aj=H;
		else if(aj<L)
			aj=L;
		if(Math.abs(alphaj-aj)<0.00001)
			return 0;
		ai = alphai + s*(alphaj-aj);
		double b1=b-ei-codelabelstrain[i]*(ai-alphai)*kernelmat[i][i]-codelabelstrain[j]*(aj-alphaj)*kernelmat[i][j];
		double b2=b-ej-codelabelstrain[i]*(ai-alphai)*kernelmat[i][j]-codelabelstrain[j]*(aj-alphaj)*kernelmat[j][j];
		if(alpha[i]<regc)
			b=b1;
		else if(alpha[j]<regc)
			b=b2;
		else
			b=(b1+b2)/2;
		alpha[i] = ai;
		alpha[j] = aj;
		//updatenonbound(j,alpha[j]);
		//updatenonbound(i,alpha[i]);
		for (int k=0; k<trainlen; k++) {
			double kik = kernelmat[i][k];  
			double kjk = kernelmat[j][k];  
			E[k] += (ai-alphai)*yi*kik + (aj-alphaj)*yj*kjk - bb + b;
		}
		return 1;
	}
	private static void SMOng(int no){
		init(no);
		int passes=0;
		int tot=0;
		while(passes<maxpass){
			tot++;
			//System.out.println(tot);
			int numalpha=0;

			for (int i = 0; i < trainlen; i++) {
				double ei=mpredict(i)-codelabelstrain[i];
				if(((codelabelstrain[i]*ei<-numtol)&&(alpha[i]<regc))||((codelabelstrain[i]*ei>numtol)&&(alpha[i]>0))){
					int j = (int)Math.floor(Math.random()*(trainlen-1));
					j = (j<i)?j:(j+1);

					numalpha+=update(i, j,ei);
				}
			}
			//System.out.println(tot+"  "+numalpha);
			if(numalpha==0)
				passes++;
			else
				passes=0;
			if(tot==500)
				break;
		}
		double acc=0.0;
		System.out.println();
		for (int i = 0; i < trainlen; i++) {
			double temp=mpredict(featurestrain[i]);
			if(temp>0)
				predictiontrain[i]=1;
			else
				predictiontrain[i]=0;
			if(temp*codelabelstrain[i]>0)
				acc++;

		}
		System.out.println("trainaga:"+acc/(double) trainlen);
		acc=0.0;
		for (int i = 0; i < testlen; i++) {
			double temp=mpredict(featurestest[i]);
			if(temp>0)
				predictiontest[i]=1;
			else
				predictiontest[i]=0;
			
			if(temp*codelabelstest[i]>0)
				acc++;

		}
		System.out.println("test:"+acc/(double) testlen);
	}
	private static void SMO_Platt(int no) {
		init(no);
		int numChanged = 0;
		int examineAll = 1;
		E = new double [trainlen];
		setcache();
		int runs=0;
		while (numChanged > 0 || examineAll == 1) {
			runs++;
			numChanged = 0;
			if (examineAll == 1) {
				for (int i=0; i<trainlen; i++) {
					//numChanged += ExamineExample(i);
					numChanged += anotherheur(i);
					//numChanged += modheur(i);
					//numChanged += newheur(i);//ExamineExample(i);
				}
			} else {
				for (int i=0; i<trainlen; i++) {
					if (alpha[i]!= 0 && alpha[i] != regc) {
						//numChanged += ExamineExample(i);
						numChanged += anotherheur(i);
						//numChanged += modheur(i);
						//numChanged += newheur(i);//ExamineExample(i);
					}
				}
			}
			if (examineAll == 1)
				examineAll = 0;
			else if (numChanged == 0)
			{
				examineAll = 1;
				double acc=0.0;
				for (int i = 0; i < trainlen; i++) {
					//System.out.println(mpredict(i)+" "+labelstrain[i]);
					if(mpredict(i)*codelabelstrain[i]>0)
						acc++;

				}
				System.out.println(acc/(double) trainlen);
			}
			//System.out.println(runs+" "+numChanged);
			if(runs>9000)
				break;
		}
		double acc=0.0;
		System.out.println();
		for (int i = 0; i < trainlen; i++) {
			double temp=mpredict(featurestrain[i]);
			if(temp>0)
				predictiontrain[i]=1;
			else
				predictiontrain[i]=0;
			if(temp*codelabelstrain[i]>0)
				acc++;

		}
		System.out.println("trainaga:"+acc/(double) trainlen);
		labelstest=changeleabels(labelstest);
		acc=0.0;
		for (int i = 0; i < testlen; i++) {
			double temp=mpredict(featurestest[i]);
			if(temp>0)
				predictiontest[i]=1;
			else
				predictiontest[i]=0;
			
			if(temp*codelabelstest[i]>0)
				acc++;

		}
		System.out.println("test:"+acc/(double) testlen);
	}
    private static double[] convert(double[][] totalpredict,int tetrlen){
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
		//setHammingCodes();
		setClassCodes(false,true);
		kernelmat=new double[trainlen][trainlen];
		setkernel();
		//setCodes();
		for (int i = 0; i < codesize; i++) {
			System.out.println(i);
			//allput(i, false, false);
			SMOng(i);
			totalpredictiontest[i]=predictiontest;
			totalpredictiontrain[i]=predictiontrain;
		}
        double[] traintp = convert(totalpredictiontrain, trainlen);
		double[] testp = convert(totalpredictiontest, testlen);
		double[] acctest = getAccuracy(testp, labelstest, testlen);
		double[] acctrain = getAccuracy(traintp, labelstrain, trainlen);
		System.out.println("Training Accuracy: "+acctrain[1]);
		System.out.println("Testing Accuracy: "+acctest[1]);
	}
}
