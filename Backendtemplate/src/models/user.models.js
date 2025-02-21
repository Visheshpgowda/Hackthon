import mongoose,{Schema} from "mongoose";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
const userSchema = new Schema({
    name:{
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    Phone_no:{
        type: String,
        required: true,
        unique: true,
    },
    password:{
        type: String,
        required: true,
    },
    role:{
        type: String,
        enum: ["Headmaster","Teacher","Parent"],
        default: "user",
    },
    refreshToken:{
        type: String,
    },
})
//used when we have hash the password when the password is modidfied 
userSchema.pre("save",async function (next){
    if(this.isModified("password"))
    {
        this.password=bcrypt.hashSync(this.password,10);
        next();
    }else{
        next();
    }
})

//used to compare the password
userSchema.methods.isPasswordcorrect=async function(password)
{
    return await bcrypt.compare(password,this.password)

}

userSchema.methods.generateAccessToken=async function(){ 
    
      return jwt.sign(
        {
            _id:this._id,
            email:this.email,
            name:this.name
        },
        process.env.ACCESS_TOKEN_SECRET,
       {expiresIn:"1d"}
      )
}
userSchema.methods.generateRefreshToken=async function(){ 

    return jwt.sign(
        {
            _id:this._id,
        },
        process.env.REFRESH_TOKEN_SECRET,
       {expiresIn:"10d"}
      )
}
export const User = mongoose.model("User", userSchema)