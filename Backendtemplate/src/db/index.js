import mongoose from "mongoose";
import { DB_NAME } from "../constants.js"

const connectionDB = async() => {
    try {
        console.log("MONGODB_URI", process.env.MONGODB_URI);
        console.log("DB_NAME", DB_NAME);
        const connectioninstance = await mongoose.connect(`${process.env.MONGODB_URI}/${DB_NAME}`)
        console.log(`\n Mongodb connected ${connectioninstance.connection.host}`)
    } catch (error) {
        console.log("MongoDB Connection Error", error)
        process.exit(1)

    }
}
export default connectionDB;