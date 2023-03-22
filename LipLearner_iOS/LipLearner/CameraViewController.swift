import AVFoundation
import UIKit

import Foundation
import Speech
import CoreML
import Accelerate
import Vision

import CreateML
import TabularData


enum workMode {
    case registeration
    case recognition
    case freeUse
}

enum recordMode {
    case command
    case keyword
    case nonSpeaking
}


final class CameraViewController: UIViewController, SFSpeechRecognizerDelegate, ItemSelectionViewControllerDelegate {
    
    @IBOutlet private weak var cameraView: UIView!
    
    @IBOutlet private weak var lipView: UIImageView!
    @IBOutlet private weak var recordButton: UIButton!
    @IBOutlet private weak var menuView: UIView!
    private var topMenuVIew: UIView!
    private var commandTableView: UITableView!
    private var tableData = [String]()
    @IBOutlet private weak var modeSelection: UISegmentedControl!
    @IBOutlet private weak var commandLabel: UILabel!
    @IBOutlet private weak var recordModeButton: UIButton!
    @IBOutlet private weak var settingButton: UIButton!
    @IBOutlet private weak var menuButton: UIButton!
    @IBOutlet private weak var KWSSwitch: UISwitch!


    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    private var captureDevice: AVCaptureDevice?
    private var captureDeviceResolution: CGSize = CGSize(width: 1080, height: 1920)
    private let videoOutput = AVCaptureVideoDataOutput()
    
    private var detectionOverlayLayer: CALayer?
    private var detectedFaceRectangleShapeLayer: CAShapeLayer?
    private var detectedFaceLandmarksShapeLayer: CAShapeLayer?
    private var detectionRequests: [VNDetectFaceRectanglesRequest]?
    private var trackingRequests: [VNTrackObjectRequest]?
    private var lipCenteredFaceBounds = CGRect(x: 403, y: 693, width: 218, height: 218)
    lazy var sequenceRequestHandler = VNSequenceRequestHandler()
    
    private var logisticRegressionClassifier: MLLogisticRegressionClassifier?
    private var KWSClassifier: MLLogisticRegressionClassifier?
    private var trainDataFrame = DataFrame()
    private var commandCenterDict = [String: [Float]]()
    private var keywordSpottingBuffer = [UIImage]()
    private var KWSWindowSize: Int = 30
    private var hopSize = 10
    private var MODQueue = Queue<Float>.init(maxCapacity: 15)
    private var MOD = Float.zero
    private var delayedMOD = Float.zero
    private var KWSDataFrame = DataFrame()
    private var keywordCandidate: [Float]?
    private var keywordSpotting: Bool = false{
        didSet{
            if keywordSpotting{
                keywordSpottingBuffer = [UIImage]()
            }
        }
    }
    
    
    private var vectorToRegister : [Float]? = nil
    private var commandToRegister : String? = nil {
        didSet{
            DispatchQueue.global(qos: .userInitiated).async {
                self.appendToDataFrame()
            }
        }
    }
    
    private var currentWorkMode = workMode.registeration
    private var currentRecordMode = recordMode.command
    private var userName: String = "User1"
    private var recording = false
    private var warmUp = true
    private var modelInput = [UIImage]()
    private let lipreadingModel = try! LipEncoder()
    private var keywordCenterVector = [Float]()
    private var keywordCount: Float = 0
    private var keywordThreshold: Float = 0.65 // Reduce this threshold to increase the sensitivity to the keyword.
    private var nonSpeakingCenterVector = [Float]()
    private var nonSpeakingCount: Float = 0
    private var nonSpeakingThreshold: Float = 0.65 // Reduce this threshold to increase the sensitivity to the end of the speech (EOS).
    
    // speech recognition related
    private var speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var recognizedText : String?
    private var languageMode = 0
    
    private var gifCount = 0
    private var freeUseCount = 0
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let deviceSessions = AVCaptureDevice.DiscoverySession(deviceTypes: [AVCaptureDevice.DeviceType.builtInWideAngleCamera], mediaType: AVMediaType.video, position: AVCaptureDevice.Position.front)

        
        for device in deviceSessions.devices {
            
            if device.hasMediaType(.video) {
                self.captureDevice = device
                if captureDevice != nil {
                    print("Capture device found")
                    beginSession()
                    break
                }
            }
        }
        
        speechRecognizer.delegate = self
        // Asynchronously make the authorization request.
        SFSpeechRecognizer.requestAuthorization { authStatus in
            print(authStatus)
        }
    }
    
    // MARK: - Actions
    override func viewDidAppear(_ animated: Bool) {
        let alert = UIAlertController(title: "Alert", message: "Input your name below", preferredStyle: .alert)
        alert.addTextField { (textField) in
            textField.placeholder = self.userName
        }
        alert.addAction(UIAlertAction(title: "New User", style: .default, handler: { [weak alert] (_) in
            self.LoadingStart()
            let textField = alert!.textFields![0] // Force unwrapping because we know it exists.
            self.userName = (textField.text == "") ? self.userName : textField.text!
            print("username:", self.userName)
            
            self.trainDataFrame.addHeader()
            self.KWSDataFrame.addHeader()
            
            self.recordButton.sendActions(for: .touchDown)
            Thread.sleep(forTimeInterval: 0.5)
            self.recordButton.sendActions(for: .touchUpInside)
        }))
        
        alert.addAction(UIAlertAction(title: "Load Data", style: .default, handler: { [weak alert] (_) in
            self.LoadingStart()
            let textField = alert!.textFields![0] // Force unwrapping because we know it exists.
            self.userName = (textField.text == "") ? self.userName : textField.text!
            print("username:", self.userName)
            
            let fileManager = FileManager.default
            do {
                let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false)
                let fileURL = path.appendingPathComponent(self.userName + ".csv")
                self.trainDataFrame = try DataFrame(contentsOfCSVFile: fileURL, options: .init(floatingPointType: .float))
                if self.trainDataFrame.rows.count == 0 {
                    self.trainDataFrame = DataFrame()
                    self.trainDataFrame.addHeader()
                }
                let kwsFileURL = path.appendingPathComponent(self.userName + "_kws.csv")
                self.KWSDataFrame = try DataFrame(contentsOfCSVFile: kwsFileURL, options: .init(floatingPointType: .float))
                print("Read csv file: \(path)")
                DispatchQueue.global(qos: .userInteractive).async{
                    self.logisticRegressionClassifier = self.trainClassifier(dataFrame: self.trainDataFrame)
                    self.KWSClassifier = self.trainClassifier(dataFrame: self.KWSDataFrame)
                }
                let datFileURL = path.appendingPathComponent(self.userName + ".dat")
                if let commandCenterDict = NSKeyedUnarchiver.unarchiveObject(withFile: datFileURL.path){
                    self.commandCenterDict = commandCenterDict as! [String : [Float]]
                    if let keywordCenterVector = self.commandCenterDict.removeValue(forKey: "keyword"){
                        self.keywordCenterVector = keywordCenterVector
                        self.keywordCount = self.keywordCenterVector.remove(at: 0)
                    }
                    if let nonSpeakingCenterVector = self.commandCenterDict.removeValue(forKey: "nonSpeaking"){
                        self.nonSpeakingCenterVector = nonSpeakingCenterVector
                        self.nonSpeakingCount = self.nonSpeakingCenterVector.remove(at: 0)
                    }
                    print("Read dat file: \(datFileURL)")
                }else{
                    print("No dat file exists!")
                }
                self.tableData = self.commandCenterDict.keys.map(){$0}.sorted(by: <)
                self.commandTableView.reloadData()
            }catch{
                print("No csv exists!")
                self.trainDataFrame.addHeader()
                self.KWSDataFrame.addHeader()
            }
            self.recordButton.sendActions(for: .touchDown)
            Thread.sleep(forTimeInterval: 0.5)
            self.recordButton.sendActions(for: .touchUpInside)
        }))
        self.present(alert, animated: true, completion: nil)
    }
    
    func appendToDataFrame(){
        guard let cmd = self.commandToRegister, let vec = self.vectorToRegister else {
            print("data is not ready")
            return
        }
        // save the command to commandDict to sort classification results by similarity
        if commandCenterDict[cmd] == nil {
            let vectorWithCount = [vec].reduce([1.0],+) // add the count 1.0 to the start of the vector
            commandCenterDict[cmd] = vectorWithCount
        }else{
            var newVector = commandCenterDict[cmd]!.suffix(from: 1).map(){$0 * commandCenterDict[cmd]![0]} // restore to sum of all nonSpeaking vectors
            newVector = zip(newVector, vec).map(+) // add new vector
            commandCenterDict[cmd]![0]+=1
            commandCenterDict[cmd]![1..<501] = newVector.map(){$0/commandCenterDict[cmd]![0]}.suffix(from: 0)
        }
        
        self.trainDataFrame.addLipData(command: cmd, vector: vec)
        //         MARK: Dubug Write to a csv file
        vectorToRegister = nil
    }
    
    func sortCommandsBySimilarity(comamndDict: [String: [Float]], vector: [Float]) -> [String]{
        var cmdSimDict = [String: Float]()
        for item in comamndDict{
            var sim: Float = .nan
            vDSP_dotpr(vector, 1, item.value, 1, &sim, vDSP_Length(500))
            var norm: Float = .nan
            vDSP_svesq(item.value, 1, &norm, vDSP_Length(500))
            cmdSimDict[item.key] = sim * norm.squareRoot()
        }
        return  cmdSimDict.sorted(by: {$0.value < $1.value}).map(){$0.key}
    }
    
    func classification(featureVector: [Float], model: MLLogisticRegressionClassifier) -> String? {
        var inputDataFrame = DataFrame()
        var inputDataDict = [String: Any]()
        for i in 0..<featureVector.count{
            inputDataFrame.append(column: Column<Float>(name: String(i), capacity: 0))
            inputDataDict[i.codingKey.stringValue] = featureVector[i]
        }
        inputDataFrame.append(valuesByColumn: inputDataDict)
        
        do{
            let result = try model.predictions(from: inputDataFrame)
            return result[0] as? String
        }catch{
            print("Classification failed")
            return nil
        }
    }
    
    func featureExtraction(modelInput: [UIImage]) -> [Float]? {
        let imgSize: Int = 88
        let imageShape: CGSize = CGSize(width: imgSize, height: imgSize)
        let mlarrayLen = modelInput.count
        if mlarrayLen >= 10 && mlarrayLen <= 128{
            let mlarray = try! MLMultiArray(shape: [1, 1, NSNumber(value: mlarrayLen), NSNumber(value: imgSize), NSNumber(value: imgSize)], dataType: MLMultiArrayDataType.float32 )
            for i in 0..<mlarrayLen {
                let imagePixel = modelInput[i].resize(to: imageShape).getPixelBuffer()
                for j in 0..<imgSize * imgSize {
                    mlarray[i * imgSize * imgSize + j] = imagePixel[j] as NSNumber
                }
            }
            let prediction = try! self.lipreadingModel.prediction(v: mlarray)
            let normedVector = prediction.var_585.withUnsafeBufferPointer(ofType: Float.self) { arrayPointer in
                var norm: Float = .nan
                vDSP_svesq(arrayPointer.baseAddress!, 1, &norm, vDSP_Length(500))
                norm = norm.squareRoot()
                var divRes = [Float](repeating: .nan, count: Int(500))
                vDSP_vsdiv(arrayPointer.baseAddress!, 1, &norm, &divRes, 1, vDSP_Length(500))
                return divRes
            }
            return normedVector
        }
        else{
            print("Input too long or too short")
            DispatchQueue.main.async {
                self.commandLabel.text = "Try again"
                if self.KWSSwitch.isOn{
                    self.keywordSpotting = true // reset the buffer because outer closure will end
                }
            }
        }
        return nil
    }
    
    @IBAction func KWSSwitchToggled(_ sender: UISwitch) {
        if sender.isOn{
            if self.nonSpeakingCount > 0 && self.keywordCount > 0{
                self.keywordSpotting = true
            }else{
                sender.setOn(false, animated: true)
                let alert = UIAlertController(title: "Alert", message: "Record keyword and nonSpeaking samples first!", preferredStyle: .alert)
                alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { _ in } ))
                self.present(alert, animated: true, completion: nil)
            }
        }else{
            self.keywordSpotting = false
        }
        
    }
    
    
    @IBAction private func recordPressed(_ sender: UIButton, forEvent event: UIEvent) {
        sender.tintColor = .red
        sender.setTitle("Recording", for: .normal)
        self.recording = true
        
        if (self.currentWorkMode == .registeration) && (self.currentRecordMode == .command) {
            do {
                try startSpeechRecognition()
                print("Starting recognition")
            } catch {
                print("Recording Not Available")
            }
        }
    }
    
    @IBAction private func recordReleased(_ sender: UIButton, forEvent event: UIEvent) {
        sender.setTitle("Start", for: .normal)
        sender.tintColor = .white
        if self.currentWorkMode != .registeration && self.keywordSpotting == false {
            self.modelInput = [UIImage]()
            self.keywordSpottingBuffer = [UIImage]()
            self.recording = false
            return
        }
        sender.tintColor = .gray
        sender.isEnabled = false
        sender.setTitle("Start", for: .normal)
        self.recording = false
        // MARK: extract feature in background
        DispatchQueue.global(qos: .userInitiated).async {
            if self.keywordSpotting {
                self.modelInput = Array(self.modelInput.prefix(upTo: self.modelInput.count - 20))
                self.keywordSpotting = false
            }
            if self.currentWorkMode == .registeration && self.currentRecordMode == .nonSpeaking {
                self.modelInput = Array(self.modelInput.prefix(upTo: min(self.modelInput.count, 30)))
            }
            if let featureVector = self.featureExtraction(modelInput: self.modelInput){
                switch self.currentWorkMode {
                case .registeration:
                    switch self.currentRecordMode{
                    case .keyword:
                        let result = self.addNewKWSVector(vector: self.keywordCenterVector, count: self.keywordCount, featureVector: featureVector)
                        self.keywordCenterVector = result.0
                        self.KWSDataFrame.addLipData(command: "P", vector: self.keywordCenterVector)
                        self.keywordCount = result.1
                    case .nonSpeaking:
                        let result = self.addNewKWSVector(vector: self.nonSpeakingCenterVector, count: self.nonSpeakingCount, featureVector: featureVector)
                        self.nonSpeakingCenterVector = result.0
                        self.KWSDataFrame.addLipData(command: "N", vector: self.nonSpeakingCenterVector)
                        self.nonSpeakingCount = result.1
                    case .command:
                        self.vectorToRegister = featureVector
                        makeGifImage(imageArray: self.modelInput, gifName: self.userName.replacingOccurrences(of: " ", with: "_") + String(self.gifCount))
                        self.gifCount += 1
                        if self.warmUp{
                            self.warmUp = false
                            self.currentRecordMode = .keyword
                            DispatchQueue.main.async {
                                self.LoadingStop()
                            }
                        }
                    }
                case .recognition:
                    //                    let tik1 = UInt64(Date().timeIntervalSince1970 * 1000) // Debug
                    guard let logisticRegressionClassifier = self.logisticRegressionClassifier, let classificationResult = self.classification(featureVector: featureVector, model: logisticRegressionClassifier) else {
                        DispatchQueue.main.async {
                            let alert = UIAlertController(title: "Alert", message: "Please train the classifier!", preferredStyle: .alert)
                            alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { _ in } ))
                            self.present(alert, animated: true, completion: nil)
                            if self.KWSSwitch.isOn{ self.keywordSpotting = true }
                            self.recordButton.tintColor = .gray
                            self.recordButton.isEnabled = true
                        }
                        return
                    }
                    makeGifImage(imageArray: self.modelInput, gifName: self.userName.replacingOccurrences(of: " ", with: "_") + classificationResult + String(self.gifCount))
                    self.gifCount+=1
                    var sortedCommands = self.sortCommandsBySimilarity(comamndDict: self.commandCenterDict, vector: featureVector)
                    sortedCommands.removeAll(where: { $0 == classificationResult })
                    sortedCommands.insert(classificationResult, at: 0)
                    //                    let tik2 = UInt64(Date().timeIntervalSince1970 * 1000) //Debug
                    //                    print("Recognition time:", tik2 - tik1 )
                    //                    print("Total time:", tik2 - tik )
                    DispatchQueue.main.async {
                        self.commandLabel.text =  classificationResult
                        let alert = UIAlertController(title: "Confirm", message: "Is the recognition result correct?", preferredStyle: .alert)
                        alert.addTextField { (textField) in
                            textField.text = classificationResult
                            let pickerView = CommandPickerView(registeredCommands: sortedCommands, frame: CGRect(x: 5, y: 50, width: 250, height: 162), textField: textField)
                            pickerView.delegate = pickerView
                            pickerView.dataSource = pickerView
                            textField.inputView = pickerView
                            alert.view.addSubview(pickerView)
                        }
                        alert.addAction(UIAlertAction(title: "Add sample", style: .default, handler: { [alert] (_) in
                            if self.KWSSwitch.isOn{
                                self.keywordSpotting = true
                            }
                            self.vectorToRegister = featureVector
                            self.commandToRegister = alert.textFields![0].text
                            guard let keywordCandidate = self.keywordCandidate else {return}
                            self.KWSDataFrame.addLipData(command: "P", vector: keywordCandidate)
                        }))
                        alert.addAction(UIAlertAction(title: "Misactivated", style: .destructive, handler: { (_) in
                            if self.KWSSwitch.isOn{
                                self.keywordSpotting = true
                            }
                            guard let keywordCandidate = self.keywordCandidate else {return}
                            DispatchQueue.global(qos: .userInteractive).async {
                                self.KWSDataFrame.addLipData(command: "N", vector: keywordCandidate)
                                self.KWSClassifier = self.trainClassifier(dataFrame: self.KWSDataFrame)
                            }
                        }))
                        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { [alert] (_) in
                            if self.KWSSwitch.isOn{
                                self.keywordSpotting = true
                            }
                        }))
                        
                        self.present(alert, animated: true, completion:  nil)
                    }
                case .freeUse:
                    self.freeUseCount += 1
                    //                    let tik1 = UInt64(Date().timeIntervalSince1970 * 1000) // Debug
                    guard let logisticRegressionClassifier = self.logisticRegressionClassifier, let classificationResult = self.classification(featureVector: featureVector, model: logisticRegressionClassifier) else {
                        DispatchQueue.main.async {
                            let alert = UIAlertController(title: "Alert", message: "Please train the classifier!", preferredStyle: .alert)
                            alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { _ in } ))
                            self.present(alert, animated: true, completion: nil)
                            if self.KWSSwitch.isOn{ self.keywordSpotting = true }
                            self.recordButton.tintColor = .gray
                            self.recordButton.isEnabled = true
                        }
                        return
                    }
                    makeGifImage(imageArray: self.modelInput, gifName: self.userName.replacingOccurrences(of: " ", with: "_") + classificationResult + String(self.gifCount))
                    self.gifCount+=1
                
                    DispatchQueue.main.async {
                        self.commandLabel.text = classificationResult
                        if self.KWSSwitch.isOn{
                            self.keywordSpotting = true
                        }
                        self.trainDataFrame.addLipData(command: classificationResult, vector: featureVector)
                        if classificationResult.range(of: "\\p{Han}", options: .regularExpression) != nil{
                            let encodedResult = classificationResult.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)
                            let shortcutURLString =  "shortcuts://run-shortcut?name=" + encodedResult!
                            UIApplication.shared.open(URL(string: shortcutURLString)!)
                        }else{
                                let shortcutURLString =  "shortcuts://run-shortcut?name=" + classificationResult.replacingOccurrences(of: " ", with: "%20")
                                UIApplication.shared.open(URL(string: shortcutURLString)!)
                        }
                    }
                    

                }
            }
            
            self.modelInput = [UIImage]()
            DispatchQueue.main.async {
                if (self.currentRecordMode != .command){
                    self.recordButton.tintColor = .white
                }else{
                    self.recordButton.tintColor = .gray
                }
                self.recordButton.isEnabled = true
            }
        }
        // MARK: stop speech recognizer and output the result to trainDataFrame
        if audioEngine.isRunning {
            audioEngine.stop()
            recognitionRequest?.endAudio()
            print("Stopping recognition")
        }
    }
    
    private func addNewKWSVector(vector: [Float], count: Float, featureVector: [Float]) -> ([Float], Float){
        if count == 0{
            return (featureVector, count+1)
        }else{
            var newVector = vector.map(){$0*count} // restore to sum of all nonSpeaking vectors
            newVector = zip(newVector, featureVector).map(+) // add new vector
            let newCount = count + 1
            newVector = newVector.map(){$0/newCount}

            return (newVector, newCount)
        }
    }
    
    @IBAction func modeChanged(_ sender: UISegmentedControl) {
        switch modeSelection.selectedSegmentIndex{
        case 0:
            currentWorkMode = .registeration
            recordModeButton.isHidden = false
            self.recordButton.tintColor = .gray
        case 1:
            currentWorkMode = .recognition
            recordModeButton.isHidden = true
            self.recordButton.tintColor = .gray
        case 2:
            currentWorkMode = .freeUse
            recordModeButton.isHidden = true
            self.recordButton.tintColor = .gray
            self.freeUseCount = 0
        default:
            currentWorkMode = .recognition
        }

    }
    
    private func setUpUI(){
        
        let bounds = view.bounds
        topMenuVIew = UIView()
        topMenuVIew.frame = CGRect(x:bounds.minX, y:bounds.minY, width: bounds.width, height: 100)
        topMenuVIew.backgroundColor = .black
        view.addSubview(topMenuVIew)
        
        commandTableView = UITableView()
        commandTableView.dataSource = self
        commandTableView.delegate = self
        commandTableView.frame = CGRect(x:bounds.minX, y:100, width: bounds.width+100, height: bounds.maxY-292)
        commandTableView.register(UITableViewCell.self, forCellReuseIdentifier: "Command Cell")
        commandTableView.isHidden = true
        view.addSubview(commandTableView)
        
        menuView.frame = CGRect(x:bounds.minX, y:bounds.maxY - 192, width: bounds.width, height: 192)
        menuView.backgroundColor = .black.withAlphaComponent(0.6)
        commandLabel.text = ""
        commandLabel.frame = CGRect(x:bounds.minX, y:100, width: bounds.width, height: 40)
        commandLabel.frame = CGRect(x:0, y: 55, width: bounds.width, height: 40)
        commandLabel.textAlignment = .center
        commandLabel.font = UIFont.systemFont(ofSize: 24)
        commandLabel.alpha = 1
        
        settingButton.frame = CGRect(x:bounds.minX + 20, y:20, width: 40, height: 40)
        settingButton.tintColor = .white
        settingButton.imageView?.contentMode = .scaleAspectFit
        settingButton.contentHorizontalAlignment = .fill
        settingButton.contentVerticalAlignment = .fill
        
        menuButton.frame = CGRect(x:bounds.maxX - 55, y:20, width: 30, height: 30)
        menuButton.tintColor = .white
        menuButton.imageView?.contentMode = .scaleAspectFit
        menuButton.contentHorizontalAlignment = .fill
        menuButton.contentVerticalAlignment = .fill
        
        let menuButtonActions = [
            UIAction(title: "Save and Train", state: .off, handler: { _ in self.saveDataAndTrain()}),
            UIAction(title: "Reset keyword", state: .off, handler: { _ in
            let alert = UIAlertController(title: "Confirm", message: "Are you sure to reset the keyword and the non-speaking data?", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default, handler: { (_) in
                self.keywordCount = 0
                self.nonSpeakingCount = 0
                self.keywordCenterVector = [Float]()
                self.nonSpeakingCenterVector = [Float]()
                self.KWSClassifier = nil
            }))
            alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { (_) in }))
            self.present(alert, animated: true, completion: nil)
        }),
            UIAction(title: "Toggle Camera View", state: .off, handler: { _ in
                if self.commandTableView.isHidden {
                    self.commandTableView.isHidden = false
                    self.previewLayer?.isHidden = true
                } else {
                    self.commandTableView.isHidden = true
                    self.previewLayer?.isHidden = false
                }
            })]
        
        menuButton.menu = UIMenu(title: "", options: .displayInline, children: menuButtonActions)
        //        menuButton.menu?.
        modeSelection.frame = CGRect(x: bounds.midX - 120, y: 30, width: 240, height: 30)
        modeSelection.selectedSegmentIndex = 0
        
        recordButton.frame = CGRect(x: bounds.midX - 35, y: bounds.maxY - 110, width: 70, height: 70)
        recordButton.layer.cornerRadius = 0.5 * recordButton.bounds.size.width
        recordButton.clipsToBounds = true
        recordButton.tintColor = .gray
        recordButton.setTitle("loading...", for: .normal)
        recordButton.isEnabled = false
        
        view.addSubview(recordButton)
        view.addSubview(commandLabel)
        view.addSubview(settingButton)
        view.addSubview(menuButton)
        
        recordModeButton.frame = CGRect(x: bounds.midX - 130, y: 0, width: 100, height: 30)
        let actions = [UIAction(title: "Keyword", state: .on, handler: {_ in self.currentRecordMode = .keyword}),
                       UIAction(title: "Non speaking", state: .off, handler: {_ in self.currentRecordMode = .nonSpeaking}),
                        UIAction(title: "Command", state: .off, handler: {
            _ in self.currentRecordMode = .command
            self.recordButton.tintColor = .gray
        })]
        
        recordModeButton.menu = UIMenu(title: "", options: .singleSelection, children: actions)
        let shapeLayer = CAShapeLayer()
        shapeLayer.strokeColor = UIColor.white.cgColor
        shapeLayer.fillColor = UIColor.clear.cgColor;
        shapeLayer.lineWidth = 4
        
        
        var path = UIBezierPath()
        path = UIBezierPath(ovalIn: CGRect(x: bounds.midX - 40, y: bounds.maxY - 115, width: 80, height: 80))
        path.lineWidth = 5
        path.stroke()
        
        shapeLayer.path = path.cgPath
        
        view.layer.addSublayer(shapeLayer)
        view.layer.borderWidth = 5
        view.layer.cornerRadius = 50
        view.layer.borderColor = UIColor(red: 1, green: 0, blue: 0, alpha: 0.0).cgColor
        
        // MARK: setup drawing layers
        
        guard let rootLayer = previewLayer else {
            print("view was not property initialized")
            return
        }
        
        let captureDeviceBounds = CGRect(x: 0,
                                         y: 0,
                                         width: captureDeviceResolution.width,
                                         height: captureDeviceResolution.height)
        
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX,
                                                     y: captureDeviceBounds.midY)
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.masksToBounds = true
        overlayLayer.anchorPoint = normalizedCenterPoint
        overlayLayer.bounds = captureDeviceBounds
        overlayLayer.position = CGPoint(x: rootLayer.frame.midX, y: rootLayer.frame.midY)
        
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"
        faceRectangleShapeLayer.bounds = captureDeviceBounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint
        faceRectangleShapeLayer.fillColor = nil
        faceRectangleShapeLayer.strokeColor = UIColor.green.withAlphaComponent(0.7).cgColor
        faceRectangleShapeLayer.lineWidth = 5
        faceRectangleShapeLayer.shadowOpacity = 0.7
        faceRectangleShapeLayer.shadowRadius = 5
        
        let faceLandmarksShapeLayer = CAShapeLayer()
        faceLandmarksShapeLayer.name = "FaceLandmarksLayer"
        faceLandmarksShapeLayer.bounds = captureDeviceBounds
        faceLandmarksShapeLayer.anchorPoint = normalizedCenterPoint
        faceLandmarksShapeLayer.position = captureDeviceBoundsCenterPoint
        faceLandmarksShapeLayer.fillColor = nil
        faceLandmarksShapeLayer.strokeColor = UIColor.red.withAlphaComponent(0.7).cgColor
        faceLandmarksShapeLayer.lineWidth = 3
        faceLandmarksShapeLayer.shadowOpacity = 0.7
        faceLandmarksShapeLayer.shadowRadius = 3
        
        overlayLayer.addSublayer(faceRectangleShapeLayer)
        faceRectangleShapeLayer.addSublayer(faceLandmarksShapeLayer)
        rootLayer.addSublayer(overlayLayer)
        
        detectionOverlayLayer = overlayLayer
        detectedFaceRectangleShapeLayer = faceRectangleShapeLayer
        detectedFaceLandmarksShapeLayer = faceLandmarksShapeLayer
        
        
        
        let scaleY = rootLayer.bounds.height / captureDeviceResolution.height
        
        //        let scaleX = rootLayer.bounds.width / self.captureDeviceResolution.width
        let affineTransform = CGAffineTransform(scaleX: scaleY, y: -scaleY)
        overlayLayer.setAffineTransform(affineTransform)
    }
    
    fileprivate func drawFaceObservations(_ faceObservations: [VNFaceObservation]) {
        guard let faceRectangleShapeLayer = detectedFaceRectangleShapeLayer,
              let faceLandmarksShapeLayer = detectedFaceLandmarksShapeLayer
        else {
            return
        }
        
        CATransaction.begin()
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectanglePath = CGMutablePath()
        let faceLandmarksPath = CGMutablePath()
        
        for faceObservation in faceObservations {
            addIndicators(to: faceRectanglePath,
                          faceLandmarksPath: faceLandmarksPath,
                          for: faceObservation)
        }
        
        faceRectangleShapeLayer.path = faceRectanglePath
        faceLandmarksShapeLayer.path = faceLandmarksPath
        
        
        CATransaction.commit()
    }
    
    
    
    fileprivate func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D, to path: CGMutablePath, applying affineTransform: CGAffineTransform, closingWhenComplete closePath: Bool) {
        let pointCount = landmarkRegion.pointCount
        if pointCount > 1 {
            let points: [CGPoint] = landmarkRegion.normalizedPoints
            path.move(to: points[0], transform: affineTransform)
            path.addLines(between: points, transform: affineTransform)
            if closePath {
                path.addLine(to: points[0], transform: affineTransform)
                path.closeSubpath()
            }
        }
    }
    
    func CGPointDistance(from: CGPoint, to: CGPoint) -> CGFloat {
        return sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y))
    }
    
    fileprivate func addIndicators(to faceRectanglePath: CGMutablePath, faceLandmarksPath: CGMutablePath, for faceObservation: VNFaceObservation) {
        let displaySize = captureDeviceResolution
        
        let faceBounds = VNImageRectForNormalizedRect(faceObservation.boundingBox, Int(displaySize.width), Int(displaySize.height))
        
        
        if let landmarks = faceObservation.landmarks {
            // Landmarks are relative to -- and normalized within --- face bounds
            let affineTransform = CGAffineTransform(translationX: faceBounds.origin.x, y: faceBounds.origin.y)
                .scaledBy(x: faceBounds.size.width, y: faceBounds.size.height)
            
            // Draw eyes, lips, and nose as closed regions.
            let closedLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.outerLips,
                landmarks.innerLips,
            ]
            let xArray = landmarks.outerLips!.normalizedPoints.map(\.x)
            let yArray = landmarks.outerLips!.normalizedPoints.map(\.y)
            let innerLips = landmarks.innerLips!.normalizedPoints
            let h = CGPointDistance(from: innerLips[1], to: innerLips[4])
            let w = CGPointDistance(from: innerLips[0], to: innerLips[2])
            self.MOD = Float((h/w))
            self.MODQueue.enqueue(self.MOD)
            if self.MODQueue.currentSize == 15 {
                self.delayedMOD  = self.MODQueue.dequeue()!
            }
            guard let minLipX = xArray.min(),
                  let maxLipX = xArray.max(),
                  let minLipY = yArray.min(),
                  let maxLipY = yArray.max() else { return }
            let midLipX = (minLipX + maxLipX) / 2
            let midLipY = (minLipY + maxLipY) / 2
            let lipMidPoint = CGPoint(x: midLipX, y: midLipY).applying(affineTransform)
            let cropSize = faceBounds.width * 0.75
            let lipCenteredFaceBounds = CGRect(x: CGFloat(lipMidPoint.x - cropSize / 2), y: CGFloat(lipMidPoint.y - cropSize / 2), width: cropSize, height: cropSize)
            self.lipCenteredFaceBounds = lipCenteredFaceBounds
        }
    }
    
    
    fileprivate func prepareVisionRequest() {
        
        //self.trackingRequests = []
        var requests = [VNTrackObjectRequest]()
        
        let faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: { (request, error) in
            
            if error != nil {
                print("FaceDetection error: \(String(describing: error)).")
            }
            
            guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
                  let results = faceDetectionRequest.results else {
                return
            }
            DispatchQueue.main.async {
                // Add the observations to the tracking list
                for observation in results {
                    let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                    requests.append(faceTrackingRequest)
                }
                self.trackingRequests = requests
            }
        })
        
        // Start with detection.  Find face, then track it.
        self.detectionRequests = [faceDetectionRequest]
        
        self.sequenceRequestHandler = VNSequenceRequestHandler()
        
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    private func beginSession() {
        do {
            let captureSession = AVCaptureSession()
            captureSession.sessionPreset = .hd1920x1080
            
            guard let captureDevice = self.captureDevice else {
                print("Could not find a capture device")
                return
            }
            try captureDevice.lockForConfiguration()
            
            captureDevice.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
            captureDevice.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: 30)
            captureDevice.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: 30)
            captureDevice.unlockForConfiguration()
            
            try captureSession.addInput(AVCaptureDeviceInput(device: captureDevice))
            
            self.previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            guard let previewLayer = self.previewLayer else {
                print("Could not create a preview layer for session")
                return
            }

            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.frame = CGRect(x: 0, y: 0, width: self.view.bounds.width, height: self.view.bounds.height)
            self.cameraView.layer.addSublayer(previewLayer)
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "AVSessionQueue", attributes: []))
            captureSession.addOutput(videoOutput)
            if #available(iOS 16.0, *) {
                captureSession.beginConfiguration()
                if captureSession.isMultitaskingCameraAccessSupported {
                    captureSession.isMultitaskingCameraAccessEnabled = true
                } else {
                    print("Multiasking camera accesss not supported!!")
                }
                captureSession.commitConfiguration()
            }
            
            DispatchQueue.global(qos: .userInitiated).async{
                captureSession.startRunning()
            }
            
            self.prepareVisionRequest()
            self.setUpUI()
            
        } catch {
            print("Could not begin a capture session (\(error))")
        }
    }
    
    private func startSpeechRecognition() throws {
        
        // Cancel the previous task if it's running.
        recognitionTask?.cancel()
        self.recognitionTask = nil
        
        // Configure the audio session for the app.
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        let inputNode = audioEngine.inputNode
        
        // Create and configure the speech recognition request.
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { fatalError("Unable to create a SFSpeechAudioBufferRecognitionRequest object") }
        recognitionRequest.shouldReportPartialResults = true
        
        // Keep speech recognition data on device
        recognitionRequest.requiresOnDeviceRecognition = false
        
        // Create a recognition task for the speech recognition session.
        // Keep a reference to the task so that it can be canceled.
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            var isFinal = false
            
            if let result = result {
                // Update the text view with the results.
                self.recognizedText = result.bestTranscription.formattedString
//                self.commandLabel.text = result.bestTranscription.formattedString
                isFinal = result.isFinal
            }
            
            if error != nil || isFinal {
                // Stop recognizing speech if there is a problem.
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                if self.warmUp { return }
                self.recognitionRequest = nil
                self.recognitionTask = nil
                let alert = UIAlertController(title: "Confirm", message: "Save this command?", preferredStyle: .alert)
                alert.addTextField { (textField) in
                    textField.text = self.recognizedText ?? ""
                }

                alert.addAction(UIAlertAction(title: "OK", style: .default, handler: { [weak alert] (_) in
                    if self.KWSSwitch.isOn{
                        self.keywordSpotting = true
                    }
                    self.recognizedText = nil
                    let textField = alert!.textFields![0]
                    guard let textFiledText = textField.text else {return}
                    self.commandToRegister = textFiledText
                    guard let keywordCandidate = self.keywordCandidate else {return}
                    self.KWSDataFrame.addLipData(command: "P", vector: keywordCandidate)
                    self.tableData.insert(self.commandToRegister!, at: 0)
                    self.commandTableView.beginUpdates()
                    self.commandTableView.insertRows(at: [IndexPath(row: 0, section: 0)], with: .automatic)
                    self.commandTableView.endUpdates()
                }))
                alert.addAction(UIAlertAction(title: "Misactivated", style: .destructive, handler: { (_) in
                    if self.KWSSwitch.isOn{
                        self.keywordSpotting = true
                    }
                    self.recognizedText = nil
                    guard let keywordCandidate = self.keywordCandidate else {return}
                    DispatchQueue.global(qos: .userInteractive).async {
                        self.KWSDataFrame.addLipData(command: "N", vector: keywordCandidate)
                        self.KWSClassifier = self.trainClassifier(dataFrame: self.KWSDataFrame)
                    }
                }))
                alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { (_) in
                    if self.KWSSwitch.isOn{
                        self.keywordSpotting = true
                    }
                    self.recognizedText = nil
                }))
                
                self.present(alert, animated: true, completion:  nil)
            }
        }
        
        
        // Configure the microphone input.
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
            self.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
    }
    
    func trainClassifier(dataFrame: DataFrame) -> MLLogisticRegressionClassifier? {
        let summary = dataFrame.summary(of: "command")
        if summary["uniqueCount"][0]! > 1 { // ensure that there are more than 1 class in the data frame
            do {
                let classifier = try MLLogisticRegressionClassifier(trainingData: dataFrame, targetColumn: "command")
                return classifier
            } catch {
                print("Unexpected error: \(error).")
                return nil
            }
        } else { return nil }
    }
    
    func keywordDetection(video: [UIImage]){
        guard let keywordVector = featureExtraction(modelInput: video) else{return}
        if !self.recording{  // not recording, need to detect keyword
            var sim: Float = .nan
            vDSP_dotpr(keywordVector, 1, self.keywordCenterVector, 1, &sim, vDSP_Length(500))
            
            var norm: Float = .nan
            vDSP_svesq(self.keywordCenterVector, 1, &norm, vDSP_Length(500))
            sim *= norm.squareRoot()
            print(sim)
            // MARK: Write the similarity log to a txt file
            if sim > self.keywordThreshold{ // otherwise the data is consifered as positive candidates.
                self.keywordCandidate = keywordVector
                if let classifier = self.KWSClassifier, let classificationResult = self.classification(featureVector: keywordVector, model: classifier) {
                    print("similarity:", sim, "classification result:", classificationResult)
                    guard classificationResult == "P" else { return }
                }
                DispatchQueue.main.async {
                    let feedbackGenerator = UIImpactFeedbackGenerator(style: .heavy)
                    feedbackGenerator.impactOccurred()
                    self.recordButton.tintColor = .red
                    self.recordButton.setTitle("Recording", for: .normal)
                    self.commandLabel.text = ""
                    self.recording = true
                }
                if (self.currentWorkMode == .registeration) && (self.currentRecordMode == .command) {
                    do {
                        try startSpeechRecognition()
                        print("Starting recognition")
                    } catch {
                        print("Recording Not Available")
                    }
                }
            }

        }else{
            if self.modelInput.count > 45 { // skip the possible nonSpeaking immediately after the keyword
                var sim: Float = .nan
                vDSP_dotpr(keywordVector, 1, self.nonSpeakingCenterVector, 1, &sim, vDSP_Length(500))
                var norm: Float = .nan
                vDSP_svesq(self.nonSpeakingCenterVector, 1, &norm, vDSP_Length(500))
                sim *= norm.squareRoot()
                
                print("Similarity: ", sim, "threshold:", self.nonSpeakingThreshold)
                if sim > self.nonSpeakingThreshold { // 1.5s elapsed after keyword feedback{
                    DispatchQueue.main.async {
                        self.recordButton.sendActions(for: .touchUpInside)
                    }
                }
            }

        }
    }
    // MARK: SFSpeechRecognizerDelegate
    
    public func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        print("speech recognizer")
    }
    
    private func saveDataAndTrain(){
        self.keywordSpotting = false
        self.LoadingStart()
        DispatchQueue.global(qos: .userInteractive).async {
            let fileManager = FileManager.default
            do {
                self.KWSClassifier = self.trainClassifier(dataFrame: self.KWSDataFrame)
                self.logisticRegressionClassifier = self.trainClassifier(dataFrame: self.trainDataFrame)
                let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false)
                let csvFileURL = path.appendingPathComponent(self.userName + ".csv")
                let kwsFileURL = path.appendingPathComponent(self.userName + "_kws.csv")
                try self.trainDataFrame.writeCSV(to: csvFileURL)
                print("Write to csv file: \(csvFileURL)")
                try self.KWSDataFrame.writeCSV(to: kwsFileURL)
                print("Write to csv file: \(kwsFileURL)")
                let datFileURL = path.appendingPathComponent(self.userName + ".dat")
                var dictData = self.commandCenterDict
                
                dictData["keyword"] = self.keywordCenterVector
                dictData["keyword"]!.insert(self.keywordCount, at: 0)
                dictData["nonSpeaking"] = self.nonSpeakingCenterVector
                dictData["nonSpeaking"]!.insert(self.nonSpeakingCount, at: 0)
                NSKeyedArchiver.archiveRootObject(dictData, toFile: datFileURL.path)
                print("Write to dat file: \(datFileURL)")

            }catch {
                print("error creating file")
            }
            

            
            print("trained a classifier with commads:\n", self.trainDataFrame.columns[0])
            DispatchQueue.main.async {
                self.LoadingStop()
                if self.KWSSwitch.isOn{ self.keywordSpotting = true }
            }
        }
    }
    
    func appendToFile(log: String, url: URL){
        if let fileUpdater = try? FileHandle(forUpdating: url) {
            fileUpdater.seekToEndOfFile()
            fileUpdater.write(log.data(using: .utf8)!)
            fileUpdater.closeFile()
        }else{
            do {
                try log.write(to: url, atomically: true, encoding: .utf8)
            }catch{
                print("failed to write log")
            }
        }
    }
    @IBAction func settingButtonTapped(_ settingButton: UIButton) {
        if self.keywordSpotting{
            self.keywordSpotting = false
            self.recordButton.sendActions(for: .touchUpInside)
        }
        let itemSelectionViewController = ItemSelectionViewController(delegate: self,
                                                                      identifier: "settings",
                                                                      selectedRow: languageMode,
                                                                      recentCommands: self.trainDataFrame.columns[0].suffix(self.freeUseCount).map(){$0 as! String},
                                                                      commandSet: self.commandCenterDict.keys.map(){$0}.sorted(by: <))
        presentItemSelectionViewController(itemSelectionViewController)
    }
    
    
    private func presentItemSelectionViewController(_ itemSelectionViewController: ItemSelectionViewController) {
        let navigationController = UINavigationController(rootViewController: itemSelectionViewController)
        navigationController.navigationBar.barTintColor = .black
        navigationController.navigationBar.tintColor = view.tintColor
        present(navigationController, animated: true, completion: nil)
    }
    
    
    func itemSelectionViewController(_ itemSelectionViewController: ItemSelectionViewController, identifier: String, reviewResults: [String?],
                                     didFinishSelectingRow selectedRow: Int) {

        let languageModes = ["en-US", "ja-JP", "zh-CN","ms-MY","fr-FR", "es-419", "vi-VN"]
        print("selected language:", languageModes[selectedRow])
        if languageMode != selectedRow{
            DispatchQueue.global(qos: .userInteractive).async {
                self.speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: languageModes[selectedRow]))!
                self.languageMode = selectedRow
                self.speechRecognizer.delegate = self
            }
        }
        DispatchQueue.global(qos: .userInteractive).async {
            if self.currentWorkMode == .freeUse {
                var tempDataFrame = DataFrame(self.trainDataFrame.prefix(self.trainDataFrame.rows.count - reviewResults.count))

                let trainCount = self.trainDataFrame.rows.count
                for i in 0..<reviewResults.count {
                    let j = reviewResults.count - i - 1
                    if let review = reviewResults[j]{
                        var newRow = self.trainDataFrame.rows[trainCount - j - 1]
                        newRow[0] = review
                        tempDataFrame.append(row: newRow)
                    }
                }
                self.freeUseCount = 0
                self.trainDataFrame = tempDataFrame
                self.logisticRegressionClassifier = self.trainClassifier(dataFrame: self.trainDataFrame)
                self.keywordSpotting = true
            }
        }
    }
    
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from _: AVCaptureConnection) {
        guard let pixelBuffer: CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        var requestHandlerOptions: [VNImageOption: AnyObject] = [:]
        
        let cameraIntrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil)
        if cameraIntrinsicData != nil {
            requestHandlerOptions[VNImageOption.cameraIntrinsics] = cameraIntrinsicData
        }
        
        
        let exifOrientation = CGImagePropertyOrientation.leftMirrored
        
        guard let requests = self.trackingRequests, !requests.isEmpty else {
            // No tracking object detected, so perform initial detection
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                            orientation: exifOrientation,
                                                            options: requestHandlerOptions)
            
            do {
                guard let detectRequests = self.detectionRequests else {
                    return
                }
                try imageRequestHandler.perform(detectRequests)
            } catch let error as NSError {
                NSLog("Failed to perform FaceRectangleRequest: %@", error)
            }
            return
        }
        
        do {
            try self.sequenceRequestHandler.perform(requests,
                                                    on: pixelBuffer,
                                                    orientation: exifOrientation)
        } catch let error as NSError {
            NSLog("Failed to perform SequenceRequest: %@", error)
        }
        
        // Setup the next round of tracking.
        var newTrackingRequests = [VNTrackObjectRequest]()
        for trackingRequest in requests {
            
            guard let results = trackingRequest.results else {
                return
            }
            
            guard let observation = results[0] as? VNDetectedObjectObservation else {
                return
            }
            
            if !trackingRequest.isLastFrame {
                if observation.confidence > 0.75 {
                    trackingRequest.inputObservation = observation
                } else {
                    trackingRequest.isLastFrame = true
                }
                newTrackingRequests.append(trackingRequest)
            }
        }
        self.trackingRequests = newTrackingRequests
        
        if newTrackingRequests.isEmpty {
            // Nothing to track, so abort.
            return
        }
        
        // Perform face landmark tracking on detected faces.
        var faceLandmarkRequests = [VNDetectFaceLandmarksRequest]()
        
        // Perform landmark detection on tracked faces.
        for trackingRequest in newTrackingRequests {
            
            let faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request, error) in
                
                if error != nil {
                    print("FaceLandmarks error: \(String(describing: error)).")
                }
                
                guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
                      let results = landmarksRequest.results else {
                    return
                }
                
                // Perform all UI updates (drawing) on the main queue, not the background queue on which this handler is being called.
                if !self.warmUp{
                    DispatchQueue.main.async {
                        self.drawFaceObservations(results)
                    }
                }
            })
            
            guard let trackingResults = trackingRequest.results else {
                return
            }
            
            guard let observation = trackingResults[0] as? VNDetectedObjectObservation else {
                return
            }
            let faceObservation = VNFaceObservation(boundingBox: observation.boundingBox)
            faceLandmarksRequest.inputFaceObservations = [faceObservation]
            
            // Continue to track detected facial landmarks.
            faceLandmarkRequests.append(faceLandmarksRequest)
            
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                            orientation: exifOrientation,
                                                            options: requestHandlerOptions)
            
            
            do {
                try imageRequestHandler.perform(faceLandmarkRequests)
            } catch let error as NSError {
                NSLog("Failed to perform FaceLandmarkRequest: %@", error)
            }
        }
        
        let sourceImage = CIImage(cvImageBuffer: pixelBuffer, options: nil)
        let transform = sourceImage.orientationTransform(for: .leftMirrored).inverted()
        let transformedBounds = self.lipCenteredFaceBounds.applying(transform)
        let croppedImage = sourceImage.cropped(to: transformedBounds)
        let context = CIContext()
        guard let tempImage = context.createCGImage(croppedImage, from: croppedImage.extent) else { return }
        let image = UIImage(cgImage: tempImage, scale:1.0, orientation: .leftMirrored)
        
        if self.keywordSpotting{
            self.keywordSpottingBuffer.append(image)
            if self.keywordSpottingBuffer.count == KWSWindowSize { // MARK: KWS window size
                let keywordSpottingBufferDuplicate = keywordSpottingBuffer.map { $0 }
                self.keywordSpottingBuffer = Array(self.keywordSpottingBuffer.suffix(self.KWSWindowSize - self.hopSize)) // MARK: KWS hop length to be shorter
                if (self.MOD >= 0.1 || self.delayedMOD >= 0.1) && (!self.recording) {
                    DispatchQueue.global(qos: .userInteractive).async {
                        self.keywordDetection(video: keywordSpottingBufferDuplicate)
                    }
                }else if (self.recording) {
                    DispatchQueue.global(qos: .userInteractive).async {
                        self.keywordDetection(video: keywordSpottingBufferDuplicate)
                    }
                }
            }
        }
        if self.recording {
            self.modelInput.append(image)
            if self.modelInput.count == 128 { // max input length is limited to 128
                DispatchQueue.main.async {
                    self.recordButton.sendActions(for: .touchUpInside)
                }
            }
        }
    }
}

// MARK: - GCDAsyncSocketDelegate

extension UIImage {
    
    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func getPixelBuffer() -> [Float]
    {
        guard let cgImage = self.cgImage else {
            return []
        }
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data
        var buf : [Float] = []
        
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel
                let r = CGFloat(pixelData[pixelInfo])
                let g = CGFloat(pixelData[pixelInfo+1])
                let b = CGFloat(pixelData[pixelInfo+2])
                
                let v: Float = floor(Float(r + g + b)/3.0)/255.0
                
                buf.append(v)
            }
        }
        return buf
    }
}

extension CameraViewController: UITableViewDataSource {

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.tableData.count
    }


    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Command Cell", for: indexPath)
        let command = self.tableData[indexPath.row]

        cell.textLabel?.text = "👄" + command
        cell.tintColor = UIColor.white
        
        return cell
    }
}

extension CameraViewController: UITableViewDelegate {

}



struct ProgressDialog {
    static var alert = UIAlertController()
    static var progressView = UIProgressView()
    static var progressPoint : Float = 0{
        didSet{
            if(progressPoint == 1){
                ProgressDialog.alert.dismiss(animated: true, completion: nil)
            }
        }
    }
}
extension CameraViewController{
    func LoadingStart(){
        ProgressDialog.alert = UIAlertController(title: nil, message: "Please wait...", preferredStyle: .alert)
        
        let loadingIndicator = UIActivityIndicatorView(frame: CGRect(x: 10, y: 5, width: 50, height: 50))
        loadingIndicator.hidesWhenStopped = true
        loadingIndicator.style = .medium
        loadingIndicator.startAnimating();
        
        ProgressDialog.alert.view.addSubview(loadingIndicator)
        present(ProgressDialog.alert, animated: true, completion: nil)
    }
    
    func LoadingStop(){
        ProgressDialog.alert.dismiss(animated: true, completion: nil)
    }
}

extension DataFrame{
    
    mutating func addHeader(){
        self.append(column: Column<String>(name: "command", capacity: 0))
        for i in 0..<500{
            self.append(column: Column<Float>(name: String(i), capacity: 0))
        }
    }
    
    mutating func addLipData(command: String, vector: [Float]){
        var rowDict: [String:Any] = [:]
        rowDict["command"] = command
        for i in 0..<vector.count{
            rowDict[i.codingKey.stringValue] = vector[i]
        }
        self.append(valuesByColumn: rowDict)
    }
}

